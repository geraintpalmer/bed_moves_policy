import numpy as np
from numba import njit

hash_weights = np.array(
    (16e11, 243e8, 9e8, 16e6, 243e3, 9e3, 256, 32, 4,
      4e11,  81e8, 3e8,  4e6,  81e3, 3e3, 128, 16, 2,
      1e11,  27e8, 1e8,  1e6,  27e3, 1e3,  64,  8, 1
    ), dtype=np.int64
)

empty_state = np.array(
    (0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0
    ), dtype=np.int32
)

max_capacities = np.array([3, 2, 2, 3, 2, 2, 1, 1, 1], dtype=np.int32)

adjacency_matrix = np.array(
    [
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int32
)

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
    return (10000 * (hash_weights * state).sum()) + (patient_type * 1000)


@njit(cache=True)
def dehash_action(action_hash):
    """
    Returns an action from the hashed action:
    abc --> (a, b, c)

    Arguments:
      + `action_hash`: a three digit integer

    Returns: a 1x3 numpt array
    """
    a3 = action_hash % 10
    a2 = (action_hash // 10) % 10
    a1 = (action_hash // 100) % 10
    return a1, a2, a3


@njit(cache=True)
def get_hash_stateaction(state, patient_type, action, hash_weights):
    """
    Returns a hashable version of the state-action pair.

    Arguments:
      + `state`: a numpy array representing the state of the system,
      + `patient_type`: an integer representing the arriving customer
           type.
      + `action`: a three digit integer representing the action.

    Returns: an integer representation of the state-action pair.
    """
    hash_state_only = get_hash_state_only(state, patient_type, hash_weights)
    return hash_state_only + action


@njit(cache=True)
def get_state_action_from_hashstate(hash_state):
    """
    Separates the action from the hash_state_only.

    Arguments:
      + `hash_state`: a full hash state representing the
          state and action

    Returns:
      + `hash_state_only`: the hash state representing the state only
      + `action`: the action
    """
    action = (hash_state % 1000)
    hash_state_only = hash_state - action
    return hash_state_only, action


@njit(cache=True)
def get_resource_use_per_time_unit(state):
    """
    Calculates the resource use for a given state per time unit

    + One FTE per block containing at least one green patient
    + One FTE per amber patient
    + One FTE per red patient

    Arguments:
      + `state` an array of 27 integers {0, 1, 2, 3} representing
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
      + `state` an array of 27 integers {0, 1, 2, 3} representing
           the state of the ward.
      + `isolation_penalty`: the numerical penalty patient per
           time unit of not being in an isolation ward.

    Returns: a numerical penalty per time unit for the given state.
    """
    return state[18:24].sum() * isolation_penalty


@njit(cache=True)
def get_move_penalty(from_block, to_block, patient_type, arriving_patient_type, move_penalties, adjacency_matrix):
    """
    Calculates the penalty for moving a patient from block to block.

    Arguments:
      + `from_block`: the block the patient was removed from
      + `to_block`: the block the patient inserted to
      + `patient_type`: the type of the patient being moved, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `arriving_patient_type`: the type of the patient arriving, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `move_penalties`: a 2x3 numpy array of penalties, where the columns
           indicate patient types, and the rows indicate if the moves are
           adjacent or not.
      + `adjacency_matrix`: an 9x9 numpy matrix with entries 1 or 0 indicating
           of the blocks are adjacent or not.

    Returns: a numerical penalty for the bed moves.
    """
    if not ((from_block == to_block) and (patient_type == arriving_patient_type)):
        adj = 1 - adjacency_matrix[from_block, to_block]
        return move_penalties[adj, patient_type]
    return 0.0

@njit(cache=True)
def insert_patient(state, patient_type, to_block):
    """
    Returns the state that results from inserting a patient.

    Arguments:
      + `state` an array of 27 integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient being inserted, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `to_block`: the block the patient inserted to

    Returns: a numpy array representing the state after the insert.
    """
    state[(patient_type * 9) + to_block] += 1
    return state


@njit(cache=True)
def move_patient(state, patient_type, to_block, from_block):
    """
    Returns the state that results from moving a patient.

    Arguments:
      + `state` an array of 27 integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient being removed, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `to_block`: the block the patient inserted to
      + `from_block`: the block the patient was removed from

    Returns: a numpy array representing the state after moving the
               patient.
    """
    state[(patient_type * 9) + to_block] += 1
    state[(patient_type * 9) + from_block] -= 1
    return state


@njit(cache=True)
def remove_patient(state, patient_type, from_block):
    """
    Returns the state that results from removing a patient.

    Arguments:
      + `state` an array of 27 integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient being removed, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `from_block`: the block the patient was removed from

    Returns: a numpy array representing the state after removing the
               patient.
    """
    state[(patient_type * 9) + from_block] -= 1
    return state

@njit(cache=True)
def deteriorate_patient(state, patient_type, block):
    """
    Returns the state that results from an Gren patient deteriorating
    into an Amber patient.

    Arguments:
      + `state` an array of 27 integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient deteriorating, either
           1: 'amber', or 0: 'green'
      + `block`: the block the deteriorating patient is

    Returns: a numpy array representing the state after the deterioration.
    """
    state[(patient_type * 9) + block] -= 1
    state[((patient_type + 1) * 9) + block] += 1
    return state

@njit(cache=True)
def get_available_insert_moves(state):
    """
    Lists all available places where a patient can be inserted.

    Arguments:
      + `state` an array of 27 integers {0, 1, 2, 3} representing
           the state of the ward.

    Returns: a list of blocks that the patient can be inserted.    
    """
    occupancy = state[0:9] + state[9:18] + state[18:27]
    return (max_capacities > occupancy).nonzero()[0]


@njit(cache=True)
def get_available_actions(state, patient_type, actions_pool):
    """
    Lists all available actions that can happend when a patient of type
    `patient_type` arrives when the ward is in state `state`.
    An action takes the form:

    (a, b, c)

    where:
      - a is the block that the new patient will be inserted into
      - b is the type of patient to moved from block a
      - c is the block that patient will move to.

    In cases where no bed moved happen, we have (a = c) and (b = patient_type).

    Arguments:
      + `state` an array of 27 integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient to move, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `actions_pool`: a pre-assigned numpy empty array of
           size 9 + (9 * 2 * 8)

    Returns: an array of actions, where each row is an integer abc, and the
             count of valid actions.
    """
    # (9 blocks to place directly) + (9 * 8 possible moves from 2 different types of patient)
    valid_count = 0
    available_blocks = get_available_insert_moves(state)
    # Case A: Direct Insert (to_block == insert_block)
    for insert_block in available_blocks:
        actions_pool[valid_count] = (100 * insert_block) + (10 * patient_type) + insert_block
        valid_count += 1
    # Case B: Bed Move (to_block != insert_block)
    for insert_block in range(9):
        for moving_patient_type in range(3):
            if moving_patient_type != patient_type and state[(moving_patient_type * 9) + insert_block] > 0:
                for to_block in available_blocks:
                    if to_block != insert_block:
                        actions_pool[valid_count] = (100 * insert_block) + (10 * moving_patient_type) + to_block
                        valid_count += 1
    return actions_pool, valid_count


@njit(cache=True)
def find_idx_of_patient_to_move(block, patient_type, patients_blocks, patients_types):
    """
    Finds the index of the patient who matches both the block and patient type.

    Arguments:
      + `block`: the block we want to match
      + `patient_type`: the patient type we want to match
      + `patients_blocks`: a numpy array of length 17 representing the blocks
            where each of the patients are
      + `patients_types`: a numpy array of length 17 representing the patient
            types of each patient

    Returns: an index where they match.
    """
    for i in range(17):
        if (patients_types[i] == patient_type) and (block == patients_blocks[i]):
            return i
    return None
