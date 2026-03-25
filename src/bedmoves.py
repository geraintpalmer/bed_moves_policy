import numpy as np
import ciw
import random
import tqdm
import pandas as pd
from collections import namedtuple
from functools import lru_cache

Qtuple = namedtuple('Qtuple', 'Q hits')
Patients = namedtuple('Patients', 'patient_types los exit_dates blocks')

class NullLock(object):
    """
    A null context to stand in for the parallel 'lock' thread context
    required for parallel processing. Used as default when not
    processing in parallel.
    """
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass

max_capacities = np.array([3, 2, 2, 3, 2, 2, 1, 1, 1])

empty_state = np.array(
    (
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0)
    ), dtype=int
)

def get_hash_state(state, action):
    """
    Returns a hashable version of the state.

    Arguments:
      + `state`: a tuple with first element a numpy array
          representing the state of the system, and second element
          an integer representing the arriving customer type.
      + `action`: the block to insert a patient.

    Returns: an integer representation of the state-action pair.
    """
    s0 = state[0][0]
    s1 = state[0][1]
    s2 = state[0][2]
    p = state[1]
    
    A = (s0[0] * 16) + (s1[0] * 4) + s1[0]
    B = (s0[1] * 243) + (s1[1] * 81) + (s2[1] * 27) + (s0[2] * 9) + (s1[2] * 3) + s2[2]
    D = (s0[3] * 16) + (s1[3] * 4) + s2[3]
    E = (s0[4] * 243) + (s1[4] * 81) + (s2[4] * 27) + (s0[5] * 9) + (s1[5] * 3) + s2[5]
    G = (s0[6] * 256) + (s1[6] * 128) + (s2[6] * 64) + (s0[7] * 32) + (s1[7] * 16) + (s2[7] * 8) + (s0[8] * 4) + (s1[8] * 2) + s2[8]

    k = A
    k = (k * 1000) + B
    k = (k * 100) + D
    k = (k * 1000) + E
    k = (k * 1000) + G
    k = (k * 10) + state[1]
    k = (k * 10) + action
    return k

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
    green = (state[0,:] > 0).sum()
    amber = state[1,:].sum()
    red = state[2,:].sum()
    return green + amber + red


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
    return state[2,:-3].sum() * isolation_penalty



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
    new_state = state.copy()
    new_state[patient_type, from_block] -= 1
    new_state[patient_type, to_block] += 1
    return new_state


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
    new_state = state.copy()
    new_state[patient_type, to_block] += 1
    return new_state


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
    new_state = state.copy()
    new_state[patient_type, from_block] -= 1
    return new_state


def get_available_insert_moves(state):
    """
    Lists all available places where a patient can be inserted.

    Arguments:
      + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
         the state of the ward.

    Returns: a list of blocks that the patient can be inserted.    
    """
    return np.flatnonzero(max_capacities - state.sum(axis=0) > 0)


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


def sort_arrays(keys, vals, hits):
    """
    Sorts the three arrays based on the `keys` array.

    Arguments
      - `keys` a numpy array of int64, the state-action pairs on which to sort
      - `vals` a numpy array of float64, the Q-values associated with the state-action pairs
      - `hits` a numpy array of int64, the number of hits per state-action pair

    Returns: the same three arrays sorted.
    """
    sort_indices = keys.argsort()
    sorted_keys = keys[sort_indices]
    sorted_vals = vals[sort_indices]
    sorted_hits = hits[sort_indices]
    return sorted_keys, sorted_vals, sorted_hits



def combine_arrays(keys_set, vals_set, hits_set):
    """
    Combines sets of keys, vals, and hits, such that the resulting
    vals are the weighted average of the vals across all sets (weighted by
    hits), and the number of hits are summed.

    Arguments
      - `keys_set` a list of numpy arrays of int64, the state-action pairs
      - `vals_set` a list of numpy array of float64, the Q-values associated with the state-action pairs
      - `hits_set` a list of numpy array of int64, the number of hits per state-action pair

    Returns: the combined list of keys, vals, and hits.
    """
    all_keys = np.concatenate(keys_set)
    all_hits = np.concatenate(hits_set)
    all_hitsvals = np.concatenate(vals_set) * all_hits
    sort_idx = all_keys.argsort()
    sorted_all_keys = all_keys[sort_idx]
    sorted_all_hits = all_hits[sort_idx]
    sorted_all_hitsvals = all_hitsvals[sort_idx]
    combined_keys, jump_indices = np.unique(sorted_all_keys, return_index=True)
    combined_hits = np.maximum(np.add.reduceat(sorted_all_hits, jump_indices), 1)
    combined_vals = np.add.reduceat(sorted_all_hitsvals, jump_indices) / combined_hits
    return combined_keys, combined_vals, combined_hits



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

    def choose_arriving_block(self, state, patient_type):
        """
        Randomly chooses a block for an arriving patient 1-epsilon
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
            return self.choose_best_block(
                state=state,
                patient_type=patient_type,
                available_blocks=available_blocks
            )
        return self.choose_random_block(
            available_blocks=available_blocks
        )

    def choose_random_block(self, available_blocks):
        """
        Chooses a block randomly from a list of blocks.

        Arguments:
          + `available_blocks`: a list of available blocks to
             insert a patient to.

        Returns: a block to place the arriving patient.
        """
        return random.choice(available_blocks)

    def choose_best_block(self, state, patient_type, available_blocks):
        """
        Chooses the best action.

        Arguments:
          + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
             the state of the ward.
          + `patient_type`: the type of the patient arriving, either
             2: 'red', 1: 'amber', or 0: 'green'
          + `available_blocks`: a list of available blocks to
             insert a patient to.

        Returns: a block to place the arriving patient.
        """
        available_blocks_Q = np.array(
            [
                self.QLearning.getQ(
                    get_hash_state(
                        state=(state, patient_type),
                        action=a
                    )
                ) for a in available_blocks
            ]
        ) + (np.random.rand(available_blocks.size) / 10e13)
        idx = available_blocks_Q.argmax()
        return available_blocks[idx]

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
          + `initial_Qvalues`: a dataframe of Q-values
          + `learn`: a Boolean, indicating if the object should carry
             out learning on this run of the simulation or not.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.transform_parameter = transform_parameter
        self.previous_cost = 0.0
        self.hash_state = None
        self.hash_state_idx = None
        self.learn = learn
        self.initialise_qvals(initial_Qvalues=initial_Qvalues)

    def __repr__(self):
        return "QLearning"

    def initialise_qvals(self, initial_Qvalues):
        """
        Initialises Qvals dataframe as either empty or with some
        previously learned Q-values.

        Arguments:
          + `initial_Qvalues`: a dataframe of Q-values
        """
        if initial_Qvalues is None:
            self.keys = np.array([], dtype='int64')
            self.qvals = np.array([], dtype='float64')
            self.hits = np.array([], dtype='int64')
        else:
            self.keys = initial_Qvalues[0]
            self.qvals = initial_Qvalues[1]
            self.hits = np.zeros(self.keys.size)

        self.newQvals = {}
        self.defaultQtuple = Qtuple(Q=0.0, hits=0)

    def attach_simulation(self, simulation):
        """
        Attaches the simulation object to the QLearning object

        Arguments:
          + `simulation`: a BedMovesSimulation object.
        """
        self.simulation = simulation

    @lru_cache(maxsize=None)
    def exists_idx(self, stateaction):
        """
        Check if `stateaction` exists in self.keys.
        """
        idx = self.keys.searchsorted(stateaction)
        if (idx < self.keys.size) and (self.keys[idx] == stateaction):
            return idx
        return None

    def getQ(self, stateaction, remember_idx=False):
        """
        Returns the Q-value for a particular state-action pair
        """
        idx = self.exists_idx(stateaction)
        if idx is not None:
            if remember_idx:
                self.hash_state_idx = idx
            return self.qvals[idx]
        if remember_idx:
            self.hash_state_idx = None
        return self.newQvals.get(stateaction, self.defaultQtuple).Q

    def store_Qval(self, stateaction, newQ):
        """
        Stores the new Qvalue
        """
        if self.hash_state_idx is not None:
            self.qvals[self.hash_state_idx] = newQ
            self.hits[self.hash_state_idx
            ] += 1
        elif stateaction in self.newQvals:
            oldhits = self.newQvals[stateaction].hits
            self.newQvals[stateaction] = Qtuple(Q=newQ, hits=oldhits+1)
        else:
            self.newQvals[stateaction] = Qtuple(Q=newQ, hits=1)


    def update_Q_values(self, next_state, next_action):
        """
        Updates the Q-values according to the Q-learning update:

        Arguments:
          + `next_state`: a tuple with first element a numpy array
              representing the state the system has just reached, and
              second element an integer representing the arriving
              customer type.
          + `next_action`: the next action that has been chosen.
        """
        if self.learn:
            cost = self.simulation.overall_cost - self.previous_cost
            self.previous_cost = self.simulation.overall_cost
            R = self.transform_cost(cost=cost)
    
            if self.hash_state is not None:
                stateaction = self.hash_state
                best_future_reward = self.get_best_future_reward(
                    next_state=next_state
                )
                oldQ = self.getQ(stateaction, remember_idx=True)
                newQ = (
                    ((1 - self.learning_rate) * oldQ)
                    + (self.learning_rate * (
                        R + (
                            self.discount_factor * best_future_reward
                        )
                    ))
                )
                self.store_Qval(stateaction, newQ)

            self.hash_state = get_hash_state(
                state=next_state,
                action=next_action
            )

    def transform_cost(self, cost):
        """
        Transforms the cost (C) into a reward (R) via: R = e^{-p * C}
        where p is the transform parameter.

        Arguments:
          + `cost` the cost since the last timestamp

        Returns: a reward.
        """
        return np.exp(-self.transform_parameter * cost)

    def get_best_future_reward(self, next_state):
        """
        Returns the maximum future reward if taking the optimal action
        when in the future state.

        Arguments:
          + `next_state`: a tuple with first element a numpy array
              representing the state the system has just reached, and
              second element an integer representing the arriving
              customer type.

        Returns: the maximum epected future reward from following the
          best actions from this state onwards.
        """
        available_as = get_available_insert_moves(state=next_state[0])
        next_hash_states = np.array([
            get_hash_state(
                state=next_state,
                action=a
            ) for a in available_as
        ])
        return max(
            self.getQ(hash_state) for hash_state in next_hash_states
        )

    def merge_qvals(self):
        """
        Updates the Qvals_df with the newly learned Qvals_dict.
        """
        keys = np.array([k for k in self.newQvals.keys()], dtype='int64')
        qvals = np.array([v.Q for v in self.newQvals.values()], dtype='float64')
        hits = np.array([v.hits for v in self.newQvals.values()], dtype='int64')
        self.keys = np.concat([self.keys, keys])
        self.qvals = np.concat([self.qvals, qvals])
        self.hits = np.concat([self.hits, hits])


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
        self.QLearning.attach_simulation(simulation=self)
        
        self.prev_now = 0.0
        self.now = 0.0
        self.overall_cost = 0.0

        self.isolation_penalty = isolation_penalty
        self.adjacent_move_penalty = adjacent_move_penalty
        self.nonadjacent_move_penalty = nonadjacent_move_penalty

        ciw.seed(seed)
        self.next_arrivals = {
            0: self.arrival_distributions[0].sample(),
            1: self.arrival_distributions[1].sample(),
            2: self.arrival_distributions[2].sample()
        }
        self.patients = Patients(
            patient_types=-np.ones(17, dtype='int64'),
            los=np.ones(17) * np.inf,
            exit_dates=np.ones(17) * np.inf,
            blocks=-np.ones(17, dtype='int64'),
        )

        self.state = empty_state

    def find_next_arrival_date(self):
        """
        Returns the next date an arrival happens and
        what type of patient that arrival will be.
        """
        next_type = min(
            self.next_arrivals,
            key=self.next_arrivals.get
        )
        return self.next_arrivals[next_type], next_type

    def find_next_exit_date(self):
        """
        Returns the next date an exit happens and what
        patient is to exit.
        """
        if np.isinf(np.min(self.patients.exit_dates)):
            return float('inf'), None

        idx = self.patients.exit_dates.argmin()
        return self.patients.exit_dates[idx], idx

    def simulate_until_max_time(
        self,
        max_time,
        lock=NullLock(),
        progress_bar=False,
        progress_bar_description=None
    ):
        """
        Simulates the ward for a given amount of time.

        Arguments:
          + `max_time`: the time to stop the simulation (positive float)
          + `lock`: a context manager used for parallel processing
          + `progress_bar`: A boolean indicating if a tqdm progress bar
               should be displayed.
          + `progress_bar_description`: The string description for the
             progress bar.
        """
        if progress_bar:
            with lock:
                self.progress_bar = tqdm.tqdm(
                    total=max_time,
                    desc=progress_bar_description
                )

        while self.now < max_time:
            next_arrival, patient_type = self.find_next_arrival_date()
            next_exit, patient_idx = self.find_next_exit_date()
            if next_arrival < next_exit:
                self.arrival(
                    next_arrival=next_arrival,
                    patient_type=patient_type
                )
            else:
                self.exit(patient_idx=patient_idx)

            if progress_bar:
                with lock:
                    remaining_time = max_time - self.progress_bar.n
                    time_increment = self.now - self.prev_now
                    self.progress_bar.update(
                        min(time_increment, remaining_time)
                    )

        if progress_bar:
            with lock:
                remaining_time = max(max_time - self.progress_bar.n, 0)
                self.progress_bar.update(remaining_time)
                self.progress_bar.close()

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
        to_block = self.action_chooser.choose_arriving_block(
            state=self.state,
            patient_type=patient_type
        )
        if to_block is not None:
            self.inflict_cost(update_time=next_arrival)
            self.prev_now = self.now
            self.now = next_arrival
            self.QLearning.update_Q_values(
                next_state=(self.state, patient_type),
                next_action=to_block
            )

            idx = self.patients.patient_types.argmin()
            self.patients.patient_types[idx] = patient_type
            self.patients.los[idx] = los
            self.patients.exit_dates[idx] = self.now + los
            self.patients.blocks[idx] = to_block

            self.state = insert_patient(
                state=self.state,
                patient_type=patient_type,
                to_block=to_block
            )

    def exit(self, patient_idx):
        """
        Removes a patient from the ward.

        Arguments:
          + `patient`: The Patient object to remove.
        """
        self.inflict_cost(update_time=self.patients.exit_dates[patient_idx])
        self.prev_now = self.now
        self.now = self.patients.exit_dates[patient_idx]
        self.state = remove_patient(
            state=self.state,
            patient_type=self.patients.patient_types[patient_idx],
            from_block=self.patients.blocks[patient_idx]
        )
        self.patients.patient_types[patient_idx] = -1
        self.patients.los[patient_idx] = np.inf
        self.patients.exit_dates[patient_idx] = np.inf
        self.patients.blocks[patient_idx] = -1


    def inflict_cost(self, update_time):
        """
        Updates the overall cost, and returns the transofrmed reward.

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

