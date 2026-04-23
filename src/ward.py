import numpy as np
from numba import njit

hash_weights = np.array(
    (16e13, 243e10, 9e10, 16e8, 243e5, 9e5, 256e2, 32e2, 4e2,
      4e13,  81e10, 3e10,  4e8,  81e5, 3e5, 128e2, 16e2, 2e2,
      1e13,  27e10, 1e10,  1e8,  27e5, 1e5,  64e2,  8e2, 1e2
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
    action = (hash_state % 10)
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
def get_move_penalty(from_block, to_block, patient_type, move_penalties, adjacency_matrix):
    """
    Calculates the penalty for moving a patient from block to block.

    Arguments:
      + `from_block`: the block the patient was removed from
      + `to_block`: the block the patient inserted to
      + `patient_type`: the type of the patient being removed, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `move_penalties`: a 2x3 numpy array of penalties, where the columns
           indicate patient types, and the rows indicate if the moves are
           adjacent or not.
      + `adjacency_matrix`: an 9x9 numpy matrix with entries 1 or 0 indicating
           of the blocks are adjacent or not.

    Returns: a numerical penalty for the bed moves.
    """
    adj = 1 - adjacency_matrix[from_block, to_block]
    return move_penalties[adj, patient_type]

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
def get_available_actions(state, patient_type):
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

    Returns: an Mx3 array, where each row is an (a, b, c) action.
    """
    # (9 blocks to place directly) + (9 * 8 possible moves from 2 different types of patient)
    max_num_actions = 9 + (9 * 2 * 8)
    actions = np.empty((max_num_actions, 3), dtype=np.int32)
    count = 0
    available_blocks = get_available_insert_moves(state)
    # Case A: Direct Insert (to_block == insert_block)
    for insert_block in available_blocks:
        actions[count, 0] = insert_block
        actions[count, 1] = patient_type
        actions[count, 2] = insert_block
        count += 1
    # Case B: Bed Move (to_block != insert_block)
    for insert_block in range(9):
        for moving_patient_type in range(3):
            if moving_patient_type != patient_type and state[(moving_patient_type * 9) + insert_block] > 0:
                for to_block in available_blocks:
                    if to_block != insert_block:
                        actions[count, 0] = insert_block
                        actions[count, 1] = moving_patient_type
                        actions[count, 2] = to_block
                        count += 1
    return actions[:count]

