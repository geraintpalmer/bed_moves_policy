import numpy as np
from numpy.lib import recfunctions as rfn
import ciw
import random
import tqdm
from math import exp
from numba import typed, types, njit, jit

hash_weights = np.array(
    (
        (16e13, 243e10, 9e10, 16e8, 243e5, 9e5, 256e2, 32e2, 4e2),
        ( 4e13,  81e10, 3e10,  4e8,  81e5, 3e5, 128e2, 16e2, 2e2),
        ( 1e13,  27e10, 1e10,  1e8,  27e5, 1e5,  64e2,  8e2, 1e2),
    ), dtype=np.int64
).ravel()


max_capacities = np.array([3, 2, 2, 3, 2, 2, 1, 1, 1], dtype=np.int32)

empty_state = np.array(
    (
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0)
    ), dtype=np.int32
).ravel()

@njit(cache=True)
def get_hash_state_only(state, patient_type, hash_weights):
    """
    Returns a hashable version of the state - not including the action.

    Arguments:
      + `state`: a numpy array representing the state of the system,
      + `patient_type`: an integer representing the arriving customer
           type.

    Returns: an integer representation of the state, with 0 placeholder
    for an action.
    """
    return (hash_weights * state).sum() + (patient_type * 10)

@njit(cache=True)
def get_hash_stateaction(state, patient_type, action, hash_weights):
    """
    Returns a hashable version of the state.

    Arguments:
      + `state`: a numpy array representing the state of the system,
      + `patient_type`: an integer representing the arriving customer
           type.
      + `action`: the block to insert a patient.

    Returns: an integer representation of the state-action pair.
    """
    return get_hash_state_only(state, patient_type, hash_weights) + action

@njit(cache=True)
def get_resource_use_per_time_unit(state):
    """
    Calculates the resource use for a given state per time unit

    + One FTE per block containing at least one green patient
    + One FTE per amber patient
    + One FTE per red patient

    Arguments:
      + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
           the state of the ward.

    Returns: and integer number of resources used per time unit.
    """
    return np.count_nonzero(state[:9]) + state[9:].sum()


@njit(cache=True, fastmath=True)
def get_penalty_per_time_unit(state, isolation_penalty):
    """
    Calculates the penalty for having isolation patients in a
    general block

    Arguments:
      + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
           the state of the ward.
      + `isolation_penalty`: the numerical penalty patient per
           time unit of not being in an isolation ward.

    Returns: a numerical penalty per time unit for the given state.
    """
    return state[18:24].sum() * isolation_penalty


def get_move_penalty(from_block, to_block, b1, b2):
    """
    Calculates the instantaneous penalty associated with a bed move.

    Arguments:
      + `from_block`: the block the patient was moved from
      + `to_block`: the block the patient is moved to
      + `b1`: the penalty for moving to an adjacent block
              (representing not a move, but a penalty for
              stretching resources across blocks)
      + `b2`: the penalty for moving to a non-adjacent block

    Returns: a numerical penalty for a bed move.
    """
    A = [
        [0, b1, b2, b1, b2, b2, b2, b2, b2],
        [b1, 0, b1, b2, b1, b2, b2, b2, b2],
        [b2, b1, 0, b2, b2, b1, b2, b2, b2],
        [b1, b2, b2, 0, b1, b2, b2, b2, b2],
        [b2, b1, b2, b1, 0, b1, b2, b2, b2],
        [b2, b2, b1, b2, b1, 0, b2, b2, b2],
        [b2, b2, b2, b2, b2, b2, 0, b2, b2],
        [b2, b2, b2, b2, b2, b2, b2, 0, b2],
        [b2, b2, b2, b2, b2, b2, b2, b2, 0]
    ]
    return A[from_block][to_block]


def move_patient(state, patient_type, from_block, to_block):
    """
    Returns the state that results from moving a patient.

    Arguments:
      + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient being moved, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `from_block`: the block the patient was moved from
      + `to_block`: the block the patient is moved to

    Returns: a numpy array representing the state after the move.
    """
    state[(patient_type * 9) + from_block] -= 1
    state[(patient_type * 9) + to_block] += 1
    return state

@njit(cache=True)
def insert_patient(state, patient_type, to_block):
    """
    Returns the state that results from inserting a patient.

    Arguments:
      + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient being moved, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `to_block`: the block the patient is moved to

    Returns: a numpy array representing the state after the insert.
    """
    state[(patient_type * 9) + to_block] += 1
    return state

@njit(cache=True)
def remove_patient(state, patient_type, from_block):
    """
    Returns the state that results from removing a patient.

    Arguments:
      + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient being moved, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `from_block`: the block the patient was moved from

    Returns: a numpy array representing the state after removing the
               patient.
    """
    state[(patient_type * 9) + from_block] -= 1
    return state


@njit(cache=True)
def get_available_insert_moves(state):
    """
    Lists all available places where a patient can be inserted.

    Arguments:
      + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
           the state of the ward.

    Returns: a list of blocks that the patient can be inserted.    
    """
    occupancy = state[0:9] + state[9:18] + state[18:27]
    return (max_capacities > occupancy).nonzero()[0]


def get_available_moves(state):
    """
    Lists all available bed moves possible.

    Arguments:
      + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
           the state of the ward.

    Returns: a list of possible moves, that is tuples (a, b, c),
               where a is the patient type, b is where they move from,
               and c is where they can move to.
    """
    available_moves = []
    available_inserts = get_available_insert_moves(state=state)
    for patient_type in range(3):
        from_blocks = np.where(state[patient_type,:] > 0)[0]
        for from_block in from_blocks:
            for to_block in available_inserts:
                if from_block != to_block:
                    available_moves.append(
                        (patient_type, from_block, to_block)
                    )
    return available_moves


@njit(cache=True)
def incremental_merge(keys1, vals1, hits1, keys2, vals2, hits2):
    """
    Merges two sorted arrays of keys, vals, and hits.

    Arguments
      - `keys`: a tuple of numpy arrays of int64, the state-action
           pairs on which to sort
      - `vals`: a tuple of numpy arrays of float64, the Q-values
           associated with the state-action pairs
      - `hits`: a tuple of numpy arrays of int64, the number of
           hits per state-action pair

    Returns: the same three arrays merged sorted.
    """
    max_len = len(keys1) + len(keys2)
    keys_n = np.empty(max_len, dtype=np.int64)
    vals_n = np.empty(max_len, dtype=np.float64)
    hits_n = np.empty(max_len, dtype=np.int64)

    idx_1 = 0
    idx_2 = 0
    idx_n = 0

    while idx_1 < len(keys1) and idx_2 < len(keys2):
        if keys1[idx_1] < keys2[idx_2]:
            keys_n[idx_n], vals_n[idx_n], hits_n[idx_n] = keys1[idx_1], vals1[idx_1], hits1[idx_1]
            idx_1 += 1
        elif keys1[idx_1] > keys2[idx_2]:
            keys_n[idx_n], vals_n[idx_n], hits_n[idx_n] = keys2[idx_2], vals2[idx_2], hits2[idx_2]
            idx_2 += 1
        else:
            sum_hits = hits1[idx_1] + hits2[idx_2]
            vals_n[idx_n] = ((vals1[idx_1] * hits1[idx_1]) + (vals2[idx_2] * hits2[idx_2])) / sum_hits
            keys_n[idx_n] = keys1[idx_1]
            hits_n[idx_n] = sum_hits
            idx_1 += 1
            idx_2 += 1
        idx_n += 1

    while idx_1 < len(keys1):
        keys_n[idx_n] = keys1[idx_1]
        vals_n[idx_n] = vals1[idx_1]
        hits_n[idx_n] = hits1[idx_1]
        idx_1 += 1
        idx_n += 1

    while idx_2 < len(keys2):
        keys_n[idx_n] = keys2[idx_2]
        vals_n[idx_n] = vals2[idx_2]
        hits_n[idx_n] = hits2[idx_2]
        idx_2 += 1
        idx_n += 1

    return keys_n[:idx_n], vals_n[:idx_n], hits_n[:idx_n]


@njit(cache=True)
def choose_random_block(available_blocks):
    """
    Chooses a block randomly from a list of blocks.

    Arguments:
      + `available_blocks`: a list of available blocks to
           insert a patient to.

    Returns: a block to place the arriving patient.
    """
    return np.random.choice(available_blocks)

@njit(cache=True)
def choose_best_block(state, patient_type, available_blocks, Qvals):
    """
    Chooses the best action.

    Arguments:
      + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient arriving, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `available_blocks`: a list of available blocks to
           insert a patient to.

    Returns: a block to place the arriving patient, and the Q-value
               associated with the state-best-action pair
    """
    hash_state_only = get_hash_state_only(
        state=state,
        patient_type=patient_type,
        hash_weights=hash_weights
    )

    available_blocks_Q = np.zeros(len(available_blocks))
    for i, a in enumerate(available_blocks):
        key = hash_state_only + a
        if key in Qvals:
            available_blocks_Q[i] = Qvals[key]

    Qs_with_rnd = (
        available_blocks_Q + (
            np.random.random(available_blocks.size) * 10e-13
        )
    )

    idx = Qs_with_rnd.argmax()
    return available_blocks[idx], available_blocks_Q[idx]


@njit(cache=True)
def get_best_future_reward(next_state, next_patient_type, Qvals, just_chose_best, prev_best_Q):
    """
    Returns the maximum future reward if taking the optimal action
    when in the future state.

    Arguments:
      + `next_state`: a numpy array representing the state the
          system has just reached
      + `next_patient_type`: an integer representing the arriving
          customer type

    Returns: the maximum epected future reward from following the
      best actions from this state onwards.
    """
    if just_chose_best:
        return prev_best_Q

    available_as = get_available_insert_moves(state=next_state)
    hash_state_only = get_hash_state_only(
        state=next_state,
        patient_type=next_patient_type,
        hash_weights=hash_weights
    )

    best_Q = 0.0
    for a in available_as:
        hash_state = hash_state_only + a
        if hash_state in Qvals:
            Q = Qvals[hash_state]
            if Q > best_Q:
                best_Q = Q

    return best_Q


@njit(cache=True, fastmath=True)
def update_Q_values(first, hash_state, next_state, next_patient_type, next_action, Qvals, hits, reward, learning_rate, discount_factor, just_chose_best, prev_best_Q):
    """
    Updates the Q-values according to the Q-learning update:

    Arguments:
      + `next_state`: a numpy array representing the state the
           system has just reached
      + `next_patient_type`: an integer representing the arriving
           customer type
      + `next_action`: the next action that has been chosen.
    """
    if not first:
        best_future_reward = get_best_future_reward(
            next_state=next_state,
            next_patient_type=next_patient_type,
            Qvals=Qvals,
            just_chose_best=just_chose_best,
            prev_best_Q=prev_best_Q
        )

        oldQ = 0.0
        oldhits = np.int64(0)
        if hash_state in Qvals:
            oldQ = Qvals[hash_state]
            oldhits = hits[hash_state]

        newQ = (
            ((1 - learning_rate) * oldQ)
            + (learning_rate * (
                reward + (
                    discount_factor * best_future_reward
                )
            ))
        )
        Qvals[hash_state] = newQ
        hits[hash_state] = oldhits + np.int64(1)

    next_hash_state = get_hash_stateaction(
        state=next_state,
        patient_type=next_patient_type,
        action=next_action,
        hash_weights=hash_weights
    )
    return next_hash_state


@njit(cache=True, fastmath=True)
def transform_cost(cost, transform_parameter):
    """
    Transforms the cost (C) into a reward (R) via: R = e^{-p * C}
    where p is the transform parameter.

    Arguments:
      + `cost` the cost since the last timestamp

    Returns: a reward.
    """
    return exp(-transform_parameter * cost)


@njit(cache=True)
def initialise_qvals(initial_Qvalues, Qvals, hits):
    """
    Initialises Qvals dictionary as either empty or with some
    previously learned Q-values. Also sets the default tuple.

    Arguments:
      + `initial_Qvalues`: a numpy structured array of Q-values
           with columns [Key, Q, Hits]
    """
    for k, v in zip(
        initial_Qvalues[0],
        initial_Qvalues[1]
    ):
        Qvals[k] = v
        hits[k] = np.int64(1)

@njit(cache=True)
def get_arrays_from_qvals(Qvals, hits):
    """
    Directly extracts arrays from the TypedDict at C-speed.
    """
    n = len(Qvals)
    keys_arr = np.empty(n, dtype=np.int64)
    q_arr = np.empty(n, dtype=np.float64)
    hits_arr = np.empty(n, dtype=np.int64)
    
    i = 0
    for k, v in Qvals.items():
        keys_arr[i] = k
        q_arr[i] = v
        hits_arr[i] = hits[k]
        i += 1
    return n, keys_arr, q_arr, hits_arr


@njit(cache=True)
def find_next_arrival_date(next_arrivals):
    """
    Returns the next date an arrival happens and
    what type of patient that arrival will be.
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
    Returns the next date an exit happens and the indec of the
    patient that is to exit.
    """
    idx = exit_dates.argmin()
    return exit_dates[idx], idx

@njit(cache=True)
def get_state_action_from_hashstate(hash_state):
    action = (hash_state % 10)
    hash_state_only = hash_state - action
    return hash_state_only, action

@njit(cache=True)
def initialise_policy(keys, vals, policy):
    for k, v in zip(keys, vals):
        hash_state_only, a = get_state_action_from_hashstate(k)
        if hash_state_only in policy:
            if policy[hash_state_only] < v:
                policy[hash_state_only] = a
        else:
            policy[hash_state_only] = a
    return policy


class EpsilonHard:
    def __init__(self, epsilon, QLearning):
        """
        Initialises the epsilon-hard action selection policy object.
        That is, chooses the best strategy `epislon` on the time, and
        randomly otherwise.

        Arguments:
          + `epsilon`: a probability, float between 0 and 1
               (low: explore more, high: exploit more)
          + `QLearning`: a QLearning object.
        """
        self.epsilon = epsilon
        self.QLearning = QLearning
        self.just_chose_best = False
        self.best_action_Q = 0.0

    def exploit_policy(self, state, patient_type, policy):
        hash_state_only = get_hash_state_only(
            state=state,
            patient_type=patient_type,
            hash_weights=hash_weights
        )
        if hash_state_only in policy:
            return policy[hash_state_only]
        available_blocks = get_available_insert_moves(state=state)
        if available_blocks.size == 0:
            return None
        return choose_random_block(available_blocks=available_blocks)

    def choose_arriving_block(self, state, patient_type):
        """
        Randomly chooses a block for an arriving patient (1-epsilon)
        of the time. Otherwise chooses the best.

        Arguments:
          + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
               the state of the ward.
          + `patient_type`: the type of the patient arriving, either
               2: 'red', 1: 'amber', or 0: 'green'

        Returns: a block to place the arriving patient.
        """
        available_blocks = get_available_insert_moves(state=state)
        if available_blocks.size == 0:
            return None
        if random.random() < self.epsilon:
            self.just_chose_best = True
            a, Qa = choose_best_block(
                state=state,
                patient_type=patient_type,
                available_blocks=available_blocks,
                Qvals=self.QLearning.Qvals
            )
            self.best_action = a
            self.best_action_Q = Qa
            return a

        self.just_chose_best = False
        return choose_random_block(
            available_blocks=available_blocks
        )

    def __repr__(self):
        """
        Returns a string representation of the object.
        """
        return f"EpsilonHard-{self.epsilon}"



class QLearning:
    def __init__(
        self,
        learning_rate,
        discount_factor,
        transform_parameter,
        initial_Qvalues=None,
        policy=None,
        learn=True
    ):
        """
        Initialises the QLearning object.

        Arguments:
          + `learning_rate`: the learning rate of the Q-learning
               algorithm (a number between 0 and 1)
          + `discount_factor`: the discount factor of the Q-learning
               algorithm (a number between 0 and 1)
          + `transform_parameter`: a parameter to transform costs into
               rewards via e^{-transform_parameter * cost}
          + `initial_Qvalues`: a numpy structured array of Q-values
               with columns [Key, Q, Hits]
          + `learn`: a Boolean, indicating if the object should carry
               out learning on this run of the simulation or not.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.transform_parameter = transform_parameter
        self.previous_cost = 0.0
        self.hash_state = np.int64(0)
        self.learn = learn
        self.Qvals = typed.Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.hits = typed.Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        if initial_Qvalues is not None:
            if self.learn:
                initialise_qvals(
                    initial_Qvalues=initial_Qvalues,
                    Qvals=self.Qvals,
                    hits=self.hits
                )
            else:
                self.policy = typed.Dict.empty(
                    key_type=types.int64,
                    value_type=types.int64
                )
                initialise_policy(
                    keys=initial_Qvalues[0],
                    vals=initial_Qvalues[1],
                    policy=self.policy
                )

    def __repr__(self):
        """
        Returns a string representation of the object.
        """
        return "QLearning"

    def return_Qvals(self):
        """
        Transforms the dictionary of Q-values into a tuple of three
        numpy arrays (keys, Qs, hits).
        """
        n, k, q, h = get_arrays_from_qvals(self.Qvals, self.hits)
        idx = np.argsort(k)
        return n, k[idx], q[idx], h[idx]


class BedMoveSimulation:
    def __init__(
        self,
        arrival_distributions,
        los_distributions,
        action_chooser,
        isolation_penalty,
        adjacent_move_penalty,
        nonadjacent_move_penalty,
        QLearning,
        seed
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
          + `action_chooser`: an object that governs the choice of
               actions.
          + `isolation_penalty`: the numerical penalty patient per time
               unit of not being in an isolation ward.
          + `adjacent_move_penalty`: the penalty for moving to an
               adjacent block (representing not a move, but a penalty
               for stretching resources across blocks)
          + `nonadjacent_move_penalty`: the penalty for moving to a
               non-adjacent block
          + `QLearning`: a QLearning object that performs the q-learning
          + `seed`: the random seed for the pseudorandom number
               generator.
        """
        self.arrival_distributions = arrival_distributions
        self.los_distributions = los_distributions
        self.action_chooser = action_chooser
        self.QLearning = QLearning

        ciw.seed(seed)
        np.random.seed(seed)
        self.next_arrivals = np.array(
            [
                self.arrival_distributions[0].sample(),
                self.arrival_distributions[1].sample(),
                self.arrival_distributions[2].sample()
            ]
        )

        self.QLearning.simulation = self
        self.prev_now = 0.0
        self.now = 0.0
        self.overall_cost = 0.0
        self.first = True
        self.isolation_penalty = isolation_penalty
        self.adjacent_move_penalty = adjacent_move_penalty
        self.nonadjacent_move_penalty = nonadjacent_move_penalty

        self.patients_patient_types = -np.ones(17, dtype='int64')
        self.patients_exit_dates = np.ones(17) * np.inf
        self.patients_blocks = -np.ones(17, dtype='int64')
        self.patients_free_indices = [i for i in range(17)]
        self.patients_number_free = 17

        self.state = empty_state.copy()

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
          + `lock`: a context manager used for parallel processing
          + `shared_progress_array`: A multiprocessing array containing
               the progress of each of the parallel trials..
          + `trial`: The number of the current trial (used for the
               multiprocessing progress bar).
        """
        if shared_progress_array is not None:
            self.update_interval = max_time / 100
            self.update_threshold = self.update_interval

        while self.now < max_time:
            next_arrival, patient_type = find_next_arrival_date(self.next_arrivals)
            if self.patients_number_free == 0:
                next_exit = float('inf')
            else:
                next_exit, patient_idx = find_next_exit_date(self.patients_exit_dates)

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
        
        if not self.QLearning.learn and self.action_chooser.epsilon == 1.0:
            to_block = self.action_chooser.exploit_policy(
                state=self.state,
                patient_type=patient_type,
                policy=self.QLearning.policy
            )
        else:
            to_block = self.action_chooser.choose_arriving_block(
                state=self.state,
                patient_type=patient_type
            )

        if to_block is not None:
            self.inflict_cost(update_time=next_arrival)
            self.now = next_arrival
            if self.QLearning.learn:
                cost = self.overall_cost - self.QLearning.previous_cost
                self.QLearning.previous_cost = self.overall_cost
                R = transform_cost(
                    cost=cost,
                    transform_parameter=self.QLearning.transform_parameter
                )

                self.QLearning.hash_state = update_Q_values(
                    first=self.first,
                    hash_state=self.QLearning.hash_state,
                    next_state=self.state,
                    next_patient_type=patient_type,
                    next_action=to_block,
                    Qvals=self.QLearning.Qvals,
                    hits=self.QLearning.hits,
                    reward=R,
                    learning_rate=self.QLearning.learning_rate,
                    discount_factor=self.QLearning.discount_factor,
                    just_chose_best=self.action_chooser.just_chose_best,
                    prev_best_Q=self.action_chooser.best_action_Q
                )
                self.first = False

            idx = self.patients_free_indices[-1]
            self.patients_patient_types[idx] = patient_type
            self.patients_exit_dates[idx] = self.now + los
            self.patients_blocks[idx] = to_block
            self.patients_free_indices.pop()
            self.patients_number_free += 1

            self.state = insert_patient(
                state=self.state,
                patient_type=patient_type,
                to_block=to_block
            )

    def exit(self, patient_idx):
        """
        Removes a patient from the ward.

        Arguments:
          + `patient_idx`: The index of the patient object to remove.
        """
        self.inflict_cost(update_time=self.patients_exit_dates[patient_idx])
        self.now = self.patients_exit_dates[patient_idx]
        self.state = remove_patient(
            state=self.state,
            patient_type=self.patients_patient_types[patient_idx],
            from_block=self.patients_blocks[patient_idx]
        )
        self.patients_patient_types[patient_idx] = -1
        self.patients_exit_dates[patient_idx] = np.inf
        self.patients_blocks[patient_idx] = -1
        self.patients_free_indices.append(patient_idx)
        self.patients_number_free -= 1

    def inflict_cost(self, update_time):
        """
        Updates the overall cost, and returns the transformed reward.

        Arguments:
          + `update_time`: the time that the cost should be inflicted.
        """
        resource_use = get_resource_use_per_time_unit(state=self.state)
        penalty = get_penalty_per_time_unit(
            state=self.state,
            isolation_penalty=self.isolation_penalty
        )
        time_since = update_time - self.now
        self.overall_cost += (time_since * (resource_use + penalty))

