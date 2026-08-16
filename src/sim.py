import numpy as np
import ciw
from numba import typed, types, njit
import ward
import chooser
import rl

def make_sampling_function(dist_info):
    """
    Makes an @njit sampling function.

    Arguments:
      - `dist_info`: a tuple containing the information of the distribution.
           The first entry is a string of the distirbution's name,
           remaining entries are the parameters, following Ciw's distirbution
           intput parameter ordering.

    Returns: a function that samples from that distribution.
    """
    if dist_info[0] == 'Exponential':
        inv_rate = 1 / dist_info[1]
        
        @njit(cache=True)
        def sample():
            return -np.log(1.0 - np.random.random()) * inv_rate

        return sample

    if dist_info[0] == 'Deterministic':
        value = dist_info[1]

        @njit(cache=True)
        def sample():
            return value

        return sample

def get_mean(dist_info):
    """
    Return the expected value of a distribution.
    """
    if dist_info[0] == 'Exponential':
        return 1 / dist_info[1]
    if dist_info[0] == 'Deterministic':
        return dist_info[1]

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
    t0 = next_arrivals[0]
    t1 = next_arrivals[1]
    t2 = next_arrivals[2]
    if t0 <= t1 and t0 <= t2:
        return t0, 0
    if t1 <= t2:
        return t1, 1
    return t2, 2


@njit(cache=True)
def find_next_activity_date(dates):
    """
    Returns the next date an activity happens and the index of the
    patient that is to participate.

    Arguments:
      + `dates`: a numpy array of length 16 representing the dates
          of activity of the patients occupying each of the 16 beds
          in the ward. An unoccupied bed will have value np.inf.

    Returns: the date of the next patient to exit, and the index
               of the patient to exit.
    """
    date = float('inf')
    idx = -1
    for bed in range(16):
        if dates[bed] < date:
            date = dates[bed]
            idx = bed
    return date, idx

@njit(cache=True)
def get_state_cost(state, update_time, prev_time, isolation_penalty):
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
        deterioration_distributions,
        improvement_distributions,
        occupancy_arrival_probs,
        isolation_penalty,
        move_penalties,
        surge_penalty,
        selection_policy,
        epsilon,
        seed,
        max_time,
        learning_rate=None,
        discount_factor=None,
        initial_keys=None,
        initial_qvals=None,
        initial_policy=None,
        warmup=0.0,
        M=None
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
          + `deterioration_distributions`: a list of Ciw distribution
               objects representing the length time it takes for a patient
               to deteriorate into the next category.
          + `isolation_penalty`: the numerical penalty patient per time
               unit of not being in an isolation ward.
          + `epsilon`: a probability, float between 0 and 1
               (low: explore more, high: exploit more)
          + `max_time`: the time to stop the simulation (positive float)
          + `learning_rate`: the learning rate of the Q-learning
               algorithm (a number between 0 and 1)
          + `discount_factor`: the discount factor of the Q-learning
               algorithm (a number between 0 and 1)
          + `seed`: the random seed for the pseudorandom number
               generator.
          + `initial_keys`: a numpy array of hashed state action pairs
          + `initial_qvals`: a numpy arrays of q-values.
          + `warmup`: when evaluating, the date at which to begin
               accumulating the cost.
          + `M`: an integer, the maximum number of state-action pairs that
               can be explored. Default is None, uses a naive estimate
               based on the arrival rates and max_time.
        """
        ciw.seed(seed)
        np.random.seed(seed)
        numba_seed(seed)

        self.arrival_distributions = tuple(make_sampling_function(dist_info) for dist_info in arrival_distributions)
        self.los_distributions = tuple(make_sampling_function(dist_info) for dist_info in los_distributions)
        self.deterioration_distributions = tuple(make_sampling_function(dist_info) for dist_info in deterioration_distributions + [('Deterministic', np.inf)])
        self.improvement_distributions = tuple(make_sampling_function(dist_info) for dist_info in [('Deterministic', np.inf)] + improvement_distributions)
        self.occupancy_arrival_probs = occupancy_arrival_probs
        self.isolation_penalty = np.float32(isolation_penalty)
        self.move_penalties = np.zeros((3, 3), dtype=np.float32)
        self.move_penalties[:2, :] = move_penalties
        self.surge_penalty = np.float32(surge_penalty)
        self.learning_rate = np.float32(learning_rate)
        self.discount_factor = np.float32(discount_factor)

        self.selection_policy = selection_policy
        self.epsilon = epsilon
        self.just_chose_best = False
        self.prev_best_Q = np.float32(0.0)

        self.next_arrivals = np.array(
            [
                self.arrival_distributions[0](),
                self.arrival_distributions[1](),
                self.arrival_distributions[2]()
            ]
        )

        self.now = 0.0
        self.overall_cost = np.float32(0.0)
        self.previous_cost = np.float32(0.0)
        self.average_reward = np.float32(0.0)
        self.variance_reward = np.float32(0.0)
        self.n_rewards = 0
        self.warmup = warmup
        self.warmup_cost = np.float32(0.0)
        self.pre_warmup = True

        self.actions_pool = np.empty(15 * 16, dtype=np.int32)
        self.q_value_pool = np.empty(15 * 16, dtype=np.float32)
        self.fixed_mask = np.empty(64, dtype=bool)

        self.patients_patient_types = -np.ones(16, dtype='int64')
        self.patients_exit_dates = np.ones(16) * np.inf
        self.patients_deterioration_dates = np.ones(16) * np.inf
        self.patients_improvement_dates = np.ones(16) * np.inf
        self.patients_blocks = -np.ones(16, dtype='int64')
        self.patients_free_indices = [i for i in range(16)]
        self.patients_number_free = 16

        self.state = ward.empty_state.copy()
        self.buffer_state = ward.empty_state.copy()
        self.hash_state = None
        self.max_time = max_time
        if M is not None:
            self.M = M
        else:
            self.M = np.ceil(2 * max_time * sum((1 / get_mean(d)) for d in arrival_distributions)).astype(np.int64)
        self.setup_qvals(initial_keys, initial_qvals, initial_policy)

    def setup_qvals(self, initial_keys, initial_qvals, initial_policy):
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
        if self.pre_warmup:
            if update_time <= self.warmup:
                self.warmup_cost += cost
            else:
                residual_cost = np.float32(cost * (
                    (self.warmup - self.now) / (update_time - self.now)
                ))
                self.warmup_cost += residual_cost
                self.pre_warmup = False


    def simulate_until_max_time(
        self,
        shared_progress_array=None,
        trial=None
    ):
        """
        Simulates the ward for a given amount of time.

        Arguments:
          + `shared_progress_array`: A multiprocessing array containing
               the progress of each of the parallel trials.
          + `trial`: The number of the current trial (used for the
               multiprocessing progress bar).
        """
        next_exit = np.inf
        next_deterioration = np.inf
        next_improvement = np.inf
        next_arrival, patient_type = find_next_arrival_date(
            next_arrivals=self.next_arrivals
        )

        if shared_progress_array is not None:
            self.update_interval = self.max_time / 100
            self.update_threshold = self.update_interval

        while self.now < self.max_time:
            if (next_arrival <= next_exit) and (next_arrival <= next_deterioration) and (next_arrival < next_improvement):
                if np.random.random() < self.occupancy_arrival_probs[16 - self.patients_number_free]:
                    self.arrival(
                        next_arrival=next_arrival,
                        patient_type=patient_type
                    )
                else:
                    interarrival = self.arrival_distributions[patient_type]()
                    self.next_arrivals[patient_type] += interarrival

                next_arrival, patient_type = find_next_arrival_date(
                    next_arrivals=self.next_arrivals
                )
                next_deterioration, deteriorating_index = find_next_activity_date(
                    dates=self.patients_deterioration_dates
                )
                next_improvement, improving_index = find_next_activity_date(
                    dates=self.patients_improvement_dates
                )
                next_exit, patient_idx = find_next_activity_date(
                    dates=self.patients_exit_dates
                )
            elif (next_deterioration <= next_exit) and (next_deterioration < next_improvement):
                self.deteriorate(patient_idx=deteriorating_index)
                next_deterioration, deteriorating_index = find_next_activity_date(
                    dates=self.patients_deterioration_dates
                )
                next_improvement, improving_index = find_next_activity_date(
                    dates=self.patients_improvement_dates
                )
            elif (next_improvement < next_exit):
                self.improve(patient_idx=improving_index)
                next_deterioration, deteriorating_index = find_next_activity_date(
                    dates=self.patients_deterioration_dates
                )
                next_improvement, improving_index = find_next_activity_date(
                    dates=self.patients_improvement_dates
                )
            else:
                self.exit(patient_idx=patient_idx)
                if patient_idx == deteriorating_index:
                    next_deterioration, deteriorating_index = find_next_activity_date(
                        dates=self.patients_deterioration_dates
                    )
                if patient_idx == improving_index:
                    next_improvement, improving_index = find_next_activity_date(
                        dates=self.patients_improvement_dates
                    )
                next_exit, patient_idx = find_next_activity_date(
                    dates=self.patients_exit_dates
                )

            if shared_progress_array is not None:
                if self.now > self.update_threshold:
                    shared_progress_array[trial] = self.update_threshold
                    self.update_threshold += self.update_interval

        if shared_progress_array is not None:
            shared_progress_array[trial] = self.max_time

    def arrival(self, next_arrival, patient_type):
        """
        Generates a patient and decides where the patient should go.

        Arguments:
          + `next_arrival`: the date of the next arrival
          + `patient_type`: the type of patient that the next arrival
               will be.
        """
        interarrival = self.arrival_distributions[patient_type]()
        self.next_arrivals[patient_type] += interarrival
        los = self.los_distributions[patient_type]()
        det = self.deterioration_distributions[patient_type]()
        imp = self.improvement_distributions[patient_type]()

        if self.patients_number_free > 0:
            a, next_hash_state, next_equivalence_idx = self.decide_action(patient_type)
            a1, a2 = ward.dehash_action(action_hash=a)

            state_cost = get_state_cost(
                state=self.state,
                update_time=next_arrival,
                prev_time=self.now,
                isolation_penalty=self.isolation_penalty
            )
            a3 = ward.find_patient_type_to_move(state=self.state, from_block=a1)
            move_cost = ward.get_move_penalty(
                from_block=a1,
                to_block=a2,
                patient_type=a3,
                arriving_patient_type=patient_type,
                move_penalties=self.move_penalties,
                surge_penalty=self.surge_penalty
            )
            cost = state_cost + move_cost

            self.overall_cost += cost
            self.accumulate_warmup_cost(
                cost=cost,
                update_time=next_arrival
            )
            self.now = next_arrival
            self.learn(patient_type, a, next_hash_state, next_equivalence_idx)

            if a1 != a2:
                move_idx = ward.find_idx_of_patient_to_move(
                    block=a1,
                    patient_type=a3,
                    patients_blocks=self.patients_blocks,
                    patients_types=self.patients_patient_types
                )
                self.patients_blocks[move_idx] = a2
                ward.move_patient(
                    state=self.state,
                    patient_type=a3,
                    to_block=a2,
                    from_block=a1
                )
                if a2 == 15:
                    self.patients_patient_types[move_idx] = -1
                    self.patients_exit_dates[move_idx] = np.inf
                    self.patients_deterioration_dates[move_idx] = np.inf
                    self.patients_improvement_dates[move_idx] = np.inf
                    self.patients_blocks[move_idx] = -1
                    self.patients_free_indices.append(move_idx)
                    self.patients_number_free += 1

            arrival_idx = self.patients_free_indices[-1]
            self.patients_patient_types[arrival_idx] = patient_type
            self.patients_exit_dates[arrival_idx] = self.now + los
            self.patients_deterioration_dates[arrival_idx] = self.now + det
            self.patients_improvement_dates[arrival_idx] = self.now + imp
            self.patients_blocks[arrival_idx] = a1
            self.patients_free_indices.pop()
            self.patients_number_free -= 1
            ward.insert_patient(
                state=self.state,
                patient_type=patient_type,
                to_block=a1
            )


    def exit(self, patient_idx):
        """
        Removes a patient from the ward.

        Arguments:
          + `patient_idx`: The index of the patient to remove.
        """
        update_time = self.patients_exit_dates[patient_idx]
        cost = get_state_cost(
            state=self.state,
            update_time=update_time,
            prev_time=self.now,
            isolation_penalty=self.isolation_penalty
        )
        self.overall_cost += cost
        self.accumulate_warmup_cost(
            cost=cost,
            update_time=update_time
        )
        self.now = update_time
        ward.remove_patient(
            state=self.state,
            patient_type=self.patients_patient_types[patient_idx],
            from_block=self.patients_blocks[patient_idx]
        )
        self.patients_patient_types[patient_idx] = -1
        self.patients_exit_dates[patient_idx] = np.inf
        self.patients_deterioration_dates[patient_idx] = np.inf
        self.patients_improvement_dates[patient_idx] = np.inf
        self.patients_blocks[patient_idx] = -1
        self.patients_free_indices.append(patient_idx)
        self.patients_number_free += 1

    def deteriorate(self, patient_idx):
        """
        Changes a patient's class.

        Arguments:
          + `patient_idx`: The index of the patient to deteriorate.
        """
        update_time = self.patients_deterioration_dates[patient_idx]
        cost = get_state_cost(
            state=self.state,
            update_time=update_time,
            prev_time=self.now,
            isolation_penalty=self.isolation_penalty
        )
        self.overall_cost += cost
        self.accumulate_warmup_cost(
            cost=cost,
            update_time=update_time
        )
        self.now = update_time
        ward.deteriorate_patient(
            state=self.state,
            patient_type=self.patients_patient_types[patient_idx],
            block=self.patients_blocks[patient_idx]
        )
        self.patients_patient_types[patient_idx] += 1
        det = self.deterioration_distributions[
            self.patients_patient_types[patient_idx]
        ]()
        imp = self.improvement_distributions[
            self.patients_patient_types[patient_idx]
        ]()
        self.patients_deterioration_dates[patient_idx] = self.now + det
        self.patients_improvement_dates[patient_idx] = self.now + imp

    def improve(self, patient_idx):
        """
        Changes a patient's class.

        Arguments:
          + `patient_idx`: The index of the patient to improve.
        """
        update_time = self.patients_improvement_dates[patient_idx]
        cost = get_state_cost(
            state=self.state,
            update_time=update_time,
            prev_time=self.now,
            isolation_penalty=self.isolation_penalty
        )
        self.overall_cost += cost
        self.accumulate_warmup_cost(
            cost=cost,
            update_time=update_time
        )
        self.now = update_time
        ward.improve_patient(
            state=self.state,
            patient_type=self.patients_patient_types[patient_idx],
            block=self.patients_blocks[patient_idx]
        )
        self.patients_patient_types[patient_idx] -= 1
        det = self.deterioration_distributions[
            self.patients_patient_types[patient_idx]
        ]()
        imp = self.improvement_distributions[
            self.patients_patient_types[patient_idx]
        ]()
        self.patients_deterioration_dates[patient_idx] = self.now + det
        self.patients_improvement_dates[patient_idx] = self.now + imp

    def learn(self, patient_type, action):
        """
        Placeholder for learning.
        """
        pass

    @property
    def pessimistic_default(self):
        """
        This is the pessemistic default value used when no
        exploration has been done, i.e. the Q-value for unexplored states.
        It corresponds to the 20% percentile of a standard normal
        distribution with mean and variance of the observed rewards.
        """
        if self.variance_reward <= 0.0:
            return self.average_reward
        stdev = self.variance_reward ** 0.5
        percentile = self.average_reward - (0.84 * stdev)
        return  percentile / (1.0 - self.discount_factor)

    def decide_action(self, patient_type):
        """
        Placeholder for deciding an action.
        """
        return None

    def return_Qvals(self):
        """
        Transforms the states, qvals, and hits arrays into
        aligned block-sorted versions, and returns them.
        """
        return rl.block_sort_arrays(
            states_array=self.states,
            qval_array=self.Qvals,
            hits_array=self.hits,
            m=self.m,
            max_idx=self.max_idx
        )



class WardTraining(WardSimulation):
    def decide_action(self, patient_type):
        """
        Decides on the action to take.

        Arguments:
          + `patient_type`: the type of patient that the next arrival
               will be.

        Returns: an action.
        """
        a, Qa, next_hash_state, next_equivalence_idx = chooser.choose_action(
            state=self.state,
            patient_type=patient_type,
            selection_policy=self.selection_policy,
            epsilon=self.epsilon,
            default_future_reward=self.pessimistic_default,
            Q_index_map=self.Q_index_map,
            qval_array=self.Qvals,
            actions_pool=self.actions_pool,
            buffer_state=self.buffer_state,
            q_value_pool=self.q_value_pool,
            fixed_mask=self.fixed_mask
        )
        self.just_chose_best = Qa is not None
        self.prev_best_Q = Qa
        return a, next_hash_state, next_equivalence_idx

    def learn(self, patient_type, action, next_hash_state, next_equivalence_idx):
        """
        Performs some Q-Learning.

        Arguments:
          + `patient_type`: the type of patient that the next arrival
               will be.
          + `action`: the action taken.
        """
        R = self.previous_cost - self.overall_cost
        self.previous_cost = self.overall_cost

        self.n_rewards += 1
        if self.n_rewards > 1:
            self.variance_reward *= ((self.n_rewards - 2) / (self.n_rewards - 1))
            self.variance_reward += ((R - self.average_reward) ** 2) / self.n_rewards
        self.average_reward += ((R - self.average_reward) / self.n_rewards)

        if self.hash_state is not None:
            self.hash_state, self.max_idx = rl.update_Q_values(
                hash_state=self.hash_state,
                next_state=self.state,
                next_patient_type=patient_type,
                next_action=action,
                next_hash_state=next_hash_state,
                next_equivalence_idx=next_equivalence_idx,
                states_array=self.states,
                qval_array=self.Qvals,
                hits_array=self.hits,
                Q_index_map=self.Q_index_map,
                max_idx=self.max_idx,
                reward=R,
                learning_rate=self.learning_rate,
                discount_factor=self.discount_factor,
                just_chose_best=self.just_chose_best,
                prev_best_Q=self.prev_best_Q,
                default_future_reward=self.pessimistic_default,
                actions_pool=self.actions_pool,
                fixed_mask=self.fixed_mask
            )
        else:
            self.hash_state, equivalence_idx = ward.get_hash_stateaction(
                state=self.state,
                patient_type=patient_type,
                action=action,
                buffer_state=self.buffer_state
            )
    
    def setup_qvals(self, initial_keys, initial_qvals, initial_policy):
        """
        Sets up the Qvals and hits dictionaries

        Arguments:
          + `initial_keys`: a numpy array of hashed stateaction pairs
          + `initial_qvals`: a numpy array of q-values
        """
        self.Q_index_map = typed.Dict.empty(
            key_type=types.int64,
            value_type=types.int32
        )
        self.states = np.zeros(self.M, dtype=np.int64)
        self.Qvals = np.zeros(self.M, dtype=np.float32)
        self.hits = np.zeros(self.M, dtype=np.int16)
        self.m = 0
        self.max_idx = 0

        if initial_keys is not None:
            self.m = len(initial_keys)
            self.max_idx = len(initial_keys)
            rl.initialise_qvals(
                initial_states_array=initial_keys,
                initial_qval_array=initial_qvals,
                states_array=self.states,
                qval_array=self.Qvals,
                hits_array=self.hits,
                Q_index_map=self.Q_index_map
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
            policy=self.policy,
            actions_pool=self.actions_pool,
            buffer_state=self.buffer_state,
            fixed_mask=self.fixed_mask,
            epsilon=self.epsilon
        )
        return a, None, None

    def learn(self, patient_type, action, next_hash_state, next_equivalence_idx):
        """
        Passes as no learning takes place.
        """
        pass

    def setup_qvals(self, initial_keys, initial_qvals, initial_policy):
        """
        Sets up the Qvals and hits dictionaries
        (when learning), or the policy (when evaluating)

        Arguments:
          + `initial_keys`: a numpy array of hashed stateaction pairs
          + `initial_qvals`: a numpy array of q-values
        """
        self.policy = typed.Dict.empty(
            key_type=types.int64,
            value_type=types.int32
        )
        if initial_keys is not None:
            rl.initialise_policy_dict(
                keys_array=initial_keys,
                policy_array=initial_policy,
                policy=self.policy
            )
