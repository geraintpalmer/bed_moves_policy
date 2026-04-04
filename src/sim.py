import numpy as np
import ciw
from numba import typed, types, njit
import ward
import chooser
import rl

@njit
def numba_seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def find_next_arrival_date(next_arrivals):
    """
    Returns the next date an arrival happens and
    what type of patient that arrival will be.

    Arguments:
      + `next_arrivals`: an mapping the index of patient types (0,
          1, or 2) to the next arrival dates for those patient types

    Returns: the date of the next arrival and the patient type.
    """
    t0, t1, t2 = next_arrivals[0], next_arrivals[1], next_arrivals[2]
    if t0 <= t1 and t0 <= t2:
        return next_arrivals[0], 0
    if t1 <= t2:
        return next_arrivals[1], 1
    return next_arrivals[2], 2


@njit(cache=True)
def find_next_exit_date(exit_dates):
    """
    Returns the next date an exit happens and the index of the
    patient that is to exit.

    Arguments:
      + `exit_dates`: a numpy array of length 17 representing
          the exit dates of the patients occupying each of the
          17 beds in the ward. An unoccupied bed will have value
          np.inf.

    Returns: the date of the next patient to exit, and the index
               of the patient to exit.
    """
    idx = exit_dates.argmin()
    return exit_dates[idx], idx

@njit(cache=True)
def get_cost(state, update_time, prev_time, isolation_penalty):
    """
    Gets the cost for the time interval between now and the last update time.

    Arguments:
      + `state`: the state that the ward has been in during the time interval
      + `update_time`: the time that the cost should be inflicted.
      + `prev_time`: the previous time the cost was inflicted
      + `isolation_penalty`: the numerical penalty patient per time unit of
           not being in an isolation ward.
    """
    resource_use = ward.get_resource_use_per_time_unit(state=state)
    penalty = ward.get_penalty_per_time_unit(
        state=state,
        isolation_penalty=isolation_penalty
    )
    interval = update_time - prev_time
    cost = (interval * (resource_use + penalty))
    return cost


class WardSimulation:
    def __init__(
        self,
        arrival_distributions,
        los_distributions,
        isolation_penalty,
        epsilon,
        seed,
        learning_rate=None,
        discount_factor=None,
        transform_parameter=None,
        initial_keys=None,
        initial_qvals=None,
        warmup=0.0
    ):
        """
        Initialises the simulation object.

        Arguments:
          + `arrival_distributions`: a list of Ciw distribution objects
               representing the inter-arrival times of the green, amber,
               and red patients.
          + `los_distributions`: a list of Ciw distribution objects
               representing the length of stay times of the green,
               amber, and red patients.
          + `isolation_penalty`: the numerical penalty patient per time
               unit of not being in an isolation ward.
          + `epsilon`: a probability, float between 0 and 1
               (low: explore more, high: exploit more)
          + `learning_rate`: the learning rate of the Q-learning
               algorithm (a number between 0 and 1)
          + `discount_factor`: the discount factor of the Q-learning
               algorithm (a number between 0 and 1)
          + `transform_parameter`: a parameter to transform costs into
               rewards via e^{-transform_parameter * cost}
          + `seed`: the random seed for the pseudorandom number
               generator.
          + `initial_keys`: a numpy array of hashed state action pairs
          + `initial_qvals`: a numpy arrays of q-values.
          + `warmup`: when evaluating, the date at which to begin
               accumulating the cost.
        """
        self.arrival_distributions = arrival_distributions
        self.los_distributions = los_distributions
        self.isolation_penalty = isolation_penalty
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.transform_parameter = transform_parameter

        self.epsilon = epsilon
        self.just_chose_best = False
        self.prev_best_Q = 0.0

        ciw.seed(seed)
        np.random.seed(seed)
        numba_seed(seed)
        self.next_arrivals = np.array(
            [
                self.arrival_distributions[0].sample(),
                self.arrival_distributions[1].sample(),
                self.arrival_distributions[2].sample()
            ]
        )

        self.now = 0.0
        self.overall_cost = 0.0
        self.previous_cost = 0.0
        self.warmup = warmup
        self.warmup_cost = 0.0
        self.pre_warmup = True

        self.patients_patient_types = -np.ones(17, dtype='int64')
        self.patients_exit_dates = np.ones(17) * np.inf
        self.patients_blocks = -np.ones(17, dtype='int64')
        self.patients_free_indices = [i for i in range(17)]
        self.patients_number_free = 17

        self.state = ward.empty_state.copy()
        self.hash_state = None
        self.setup_qvals(initial_keys, initial_qvals)

    def setup_qvals(self, initial_keys, initial_qvals):
        """
        Placeholder for setting up qvals or policy.
        """
        pass

    def accumulate_warmup_cost(self, cost, update_time):
        """
        Accumulates the cost incurred during the warmup time

        Arguments:
          + `cost`: the cost incurred during the last interval
          + `update_time`: the date of the end of the interval
        """
        if update_time <= self.warmup:
            self.warmup_cost += cost
        if (update_time > self.warmup) and self.pre_warmup:
            residual_cost = cost * (
                (update_time - self.now) / (update_time - self.warmup)
            )
            self.warmup_cost += residual_cost
            self.pre_warmup = False

    def simulate_until_max_time(
        self,
        max_time,
        shared_progress_array=None,
        trial=None
    ):
        """
        Simulates the ward for a given amount of time.

        Arguments:
          + `max_time`: the time to stop the simulation (positive float)
          + `shared_progress_array`: A multiprocessing array containing
               the progress of each of the parallel trials.
          + `trial`: The number of the current trial (used for the
               multiprocessing progress bar).
        """
        if shared_progress_array is not None:
            self.update_interval = max_time / 100
            self.update_threshold = self.update_interval

        while self.now < max_time:
            next_arrival, patient_type = find_next_arrival_date(
                next_arrivals=self.next_arrivals
            )
            if self.patients_number_free == 0:
                next_exit = float('inf')
            else:
                next_exit, patient_idx = find_next_exit_date(
                    exit_dates=self.patients_exit_dates
                )

            if next_arrival < next_exit:
                self.arrival(
                    next_arrival=next_arrival,
                    patient_type=patient_type
                )
            else:
                self.exit(patient_idx=patient_idx)

            if shared_progress_array is not None:
                if self.now > self.update_threshold:
                    shared_progress_array[trial] = self.update_threshold
                    self.update_threshold += self.update_interval

        if shared_progress_array is not None:
            shared_progress_array[trial] = max_time

    def arrival(self, next_arrival, patient_type):
        """
        Generates a patient and decides where the patient should go.

        Arguments:
          + `next_arrival`: the date of the next arrival
          + `patient_type`: the type of patient that the next arrival
               will be.
        """
        interarrival = self.arrival_distributions[patient_type].sample()
        self.next_arrivals[patient_type] += interarrival
        los = self.los_distributions[patient_type].sample()
        a = self.decide_action(patient_type)

        if a is not None:
            cost = get_cost(
                state=self.state,
                update_time=next_arrival,
                prev_time=self.now,
                isolation_penalty=self.isolation_penalty
            )
            self.overall_cost += cost
            self.accumulate_warmup_cost(
                cost=cost,
                update_time=next_arrival
            )
            self.now = next_arrival
            self.learn(patient_type, a)
            idx = self.patients_free_indices[-1]
            self.patients_patient_types[idx] = patient_type
            self.patients_exit_dates[idx] = self.now + los
            self.patients_blocks[idx] = a
            self.patients_free_indices.pop()
            self.patients_number_free += 1

            self.state = ward.insert_patient(
                state=self.state,
                patient_type=patient_type,
                to_block=a
            )

    def exit(self, patient_idx):
        """
        Removes a patient from the ward.

        Arguments:
          + `patient_idx`: The index of the patient object to remove.
        """
        cost = get_cost(
            state=self.state,
            update_time=self.patients_exit_dates[patient_idx],
            prev_time=self.now,
            isolation_penalty=self.isolation_penalty
        )
        self.overall_cost += cost
        self.accumulate_warmup_cost(
            cost=cost,
            update_time=self.patients_exit_dates[patient_idx]
        )
        self.now = self.patients_exit_dates[patient_idx]
        self.state = ward.remove_patient(
            state=self.state,
            patient_type=self.patients_patient_types[patient_idx],
            from_block=self.patients_blocks[patient_idx]
        )
        self.patients_patient_types[patient_idx] = -1
        self.patients_exit_dates[patient_idx] = np.inf
        self.patients_blocks[patient_idx] = -1
        self.patients_free_indices.append(patient_idx)
        self.patients_number_free -= 1

    def learn(self, patient_type, action):
        """
        Placeholder for learning.
        """
        pass

    def decide_action(self, patient_type):
        """
        Placeholder for deciding an action.
        """
        return None

    def return_Qvals(self):
        """
        Transforms the dictionary of Q-values into a tuple of three
        numpy arrays (keys, Qs, hits).
        """
        n, k, q, h = rl.get_arrays_from_dicts(self.Qvals, self.hits)
        idx = np.argsort(k)
        return n, k[idx], q[idx], h[idx]



class WardTraining(WardSimulation):
    def decide_action(self, patient_type):
        """
        Decides on the action to take.

        Arguments:
          + `patient_type`: the type of patient that the next arrival
               will be.

        Returns: an action.
        """
        a, Qa = chooser.choose_arriving_block(
            state=self.state,
            patient_type=patient_type,
            epsilon=self.epsilon,
            Qvals=self.Qvals
        )
        self.just_chose_best = Qa is not None
        self.prev_best_Q = Qa
        return a

    def learn(self, patient_type, action):
        """
        Performs some Q-Learning.

        Arguments:
          + `patient_type`: the type of patient that the next arrival
               will be.
          + `action`: the action taken.
        """
        cost = self.overall_cost - self.previous_cost
        self.previous_cost = self.overall_cost
        R = rl.transform_cost(
            cost=cost,
            transform_parameter=self.transform_parameter
        )

        if self.hash_state is not None:
            self.hash_state = rl.update_Q_values(
                hash_state=self.hash_state,
                next_state=self.state,
                next_patient_type=patient_type,
                next_action=action,
                Qvals=self.Qvals,
                hits=self.hits,
                reward=R,
                learning_rate=self.learning_rate,
                discount_factor=self.discount_factor,
                just_chose_best=self.just_chose_best,
                prev_best_Q=self.prev_best_Q
            )
        else:
            self.hash_state = ward.get_hash_stateaction(
                state=self.state,
                patient_type=patient_type,
                action=action,
                hash_weights=ward.hash_weights
            )
    def setup_qvals(self, initial_keys, initial_qvals):
        """
        Sets up the Qvals and hits dictionaries

        Arguments:
          + `initial_keys`: a numpy array of hashed stateaction pairs
          + `initial_qvals`: a numpy array of q-values
        """
        self.Qvals = typed.Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.hits = typed.Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        if initial_keys is not None:
            rl.initialise_qvals(
                keys_array=initial_keys,
                qval_array=initial_qvals,
                Qvals=self.Qvals,
                hits=self.hits
            )

class WardEvaluation(WardSimulation):
    def decide_action(self, patient_type):
        """
        Decides on the action to take by exploting the given policy.

        Arguments:
          + `patient_type`: the type of patient that the next arrival
               will be.

        Returns: an action.
        """
        a = chooser.exploit_policy(
            state=self.state,
            patient_type=patient_type,
            policy=self.policy
        )
        return a

    def learn(self, patient_type, action):
        """
        Passes as no learning takes place.
        """
        pass

    def setup_qvals(self, initial_keys, initial_qvals):
        """
        Sets up the Qvals and hits dictionaries
        (when learning), or the policy (when evaluating)

        Arguments:
          + `initial_keys`: a numpy array of hashed stateaction pairs
          + `initial_qvals`: a numpy array of q-values
        """
        self.policy = typed.Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        if initial_keys is not None:
            rl.initialise_policy(
                keys_array=initial_keys,
                qval_array=initial_qvals,
                policy=self.policy
            )
