import numpy as np
from numba import njit

hash_weights = np.array(
    (3 * 19 * (4 ** 14), 3 * 19 * (4 ** 13), 3 * 19 * (4 ** 12), 3 * 19 * (4 ** 11), 3 * 19 * (4 ** 10), 3 * 19 * (4 ** 9), 3 * 19 * (4 ** 8), 3 * 19 * (4 ** 7), 3 * 19 * (4 ** 6), 3 * 19 * (4 ** 5), 3 * 19 * (4 ** 4), 3 * 19 * (4 ** 3), 3 * 19 * (4 ** 2), 3 * 19 * (4 ** 1), 3 * 19, 9,
     2 * 19 * (4 ** 14), 2 * 19 * (4 ** 13), 2 * 19 * (4 ** 12), 2 * 19 * (4 ** 11), 2 * 19 * (4 ** 10), 2 * 19 * (4 ** 9), 2 * 19 * (4 ** 8), 2 * 19 * (4 ** 7), 2 * 19 * (4 ** 6), 2 * 19 * (4 ** 5), 2 * 19 * (4 ** 4), 2 * 19 * (4 ** 3), 2 * 19 * (4 ** 2), 2 * 19 * (4 ** 1), 2 * 19, 3,
     1 * 19 * (4 ** 14), 1 * 19 * (4 ** 13), 1 * 19 * (4 ** 12), 1 * 19 * (4 ** 11), 1 * 19 * (4 ** 10), 1 * 19 * (4 ** 9), 1 * 19 * (4 ** 8), 1 * 19 * (4 ** 7), 1 * 19 * (4 ** 6), 1 * 19 * (4 ** 5), 1 * 19 * (4 ** 4), 1 * 19 * (4 ** 3), 1 * 19 * (4 ** 2), 1 * 19 * (4 ** 1), 1 * 19, 1
    ), dtype=np.int64
)

empty_state = np.array(
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ), dtype=np.int32
)

max_capacities = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=np.int32)

adjacency_matrix = np.array(
    [
        [2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        [0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        [0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        [0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        [0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        [0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        [0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        [0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 1, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 0, 0, 0, 1, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 0, 0, 0, 1, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 0, 1, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 1, 3],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3]
    ], dtype=np.int32
)

tiling_8 = np.array([0, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 3, 2, 3, 1, 2, 2, 2, 2, 3, 2, 3, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 3, 4, 2, 3, 3, 3, 3, 4, 3, 4, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 3, 4, 2, 3, 3, 3, 3, 4, 3, 4, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 3, 4, 2, 3, 3, 3, 3, 4, 3, 4, 2, 3, 3, 3, 3, 4, 3, 4, 3, 4, 4, 4, 3, 4, 4, 4, 2, 3, 3, 3, 3, 4, 3, 4, 3, 4, 4, 4, 3, 4, 4, 4, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 3, 4, 2, 3, 3, 3, 3, 4, 3, 4, 2, 3, 3, 3, 3, 4, 3, 4, 3, 4, 4, 4, 3, 4, 4, 4, 2, 3, 3, 3, 3, 4, 3, 4, 3, 4, 4, 4, 3, 4, 4, 4])
tiling_7 = np.array([0, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 3, 2, 3, 1, 2, 2, 2, 2, 3, 2, 3, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 3, 4, 2, 3, 3, 3, 3, 4, 3, 4, 1, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 3, 4, 2, 3, 3, 3, 3, 4, 3, 4])

@njit(cache=True)
def get_hash_state_only(state, patient_type):
    """
    Returns a hashable version of the state - not including the action.

    Arguments:
      + `state`: a numpy array representing the state of the system,
      + `patient_type`: an integer representing the arriving customer
           type.
      + `hash_weights`: the array of weights that convert the state to
           a hash via a dot product.

    Returns: an integer representation of the state, with 0 placeholder
    for an action.
    """
    return (100000 * (hash_weights * state).sum()) + (patient_type * 10000)


@njit(cache=True)
def dehash_action(action_hash):
    """
    Returns an action from the hashed action:
    abcd --> (ab, cd)

    Arguments:
      + `action_hash`: a three digit integer

    Returns: a tuple
    """
    a2 = action_hash % 100
    a1 = (action_hash // 100) % 10
    return a1, a2


@njit(cache=True)
def get_hash_stateaction(state, patient_type, action):
    """
    Returns a hashable version of the state-action pair.

    Arguments:
      + `state`: a numpy array representing the state of the system,
      + `patient_type`: an integer representing the arriving customer
           type.
      + `action`: a three digit integer representing the action.
      + `hash_weights`: the array of weights that convert the state to
           a hash via a dot product.

    Returns: an integer representation of the state-action pair.
    """
    hash_state_only = get_hash_state_only(state, patient_type)
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
    action = (hash_state % 10000)
    hash_state_only = hash_state - action
    return hash_state_only, action


@njit(cache=True)
def get_stage2_staffing(state_row):
    """
    Gets the staffing requirements for Stage 2 patients only.
    Converts a 7 and 8 bed ward configuration into integers,
    and looks up the staffing tiling resource requirement

    Arguments:
      + `state`: a numpy array representing the state of the system

    Returns: an integer number of staff required.
    """
    idx8 = 0
    for i in range(8):
        idx8 += state_row[7 - i] << i
    idx7 = 0
    for i in range(7):
        idx7 += state_row[14 - i] << i
    return tiling_8[idx8] + tiling_7[idx7]


@njit(cache=True)
def get_resource_use_per_time_unit(state):
    """
    Calculates the resource use for a given state per time unit

    + One FTE per block containing at least one green patient
    + One FTE per amber patient
    + One FTE per red patient

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.

    Returns: and integer number of resources used per time unit.
    """
    return get_stage2_staffing(state[:15]) + state[15:].sum()


@njit(cache=True, fastmath=True)
def get_penalty_per_time_unit(state, isolation_penalty):
    """
    Calculates the penalty for having isolation patients in a
    general block

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.
      + `isolation_penalty`: the numerical penalty patient per
           time unit of not being in an isolation ward.

    Returns: a numerical penalty per time unit for the given state.
    """
    return state[32:47].sum() * isolation_penalty


@njit(cache=True)
def get_move_penalty(
    from_block,
    to_block,
    patient_type,
    arriving_patient_type,
    move_penalties,
    surge_penalty
):
    """
    Calculates the penalty for moving a patient from block to block.

    Arguments:
      + `from_block`: the block the patient was removed from
      + `to_block`: the block the patient inserted to
      + `patient_type`: the type of the patient being moved, either
           2: 'Stage 3-I', 1: 'Stage 3', or 0: 'Stage 2'
      + `arriving_patient_type`: the type of the patient arriving, either
           2: 'Stage 3-I', 1: 'Stage 3', or 0: 'Stage 2'
      + `move_penalties`: a 2x3 numpy array of penalties, where the columns
           indicate patient types, and the rows indicate if the moves are
           adjacent or not.

    Returns: a numerical penalty for the bed moves.
    """
    adj = adjacency_matrix[from_block, to_block]
    if adj == 3:
        return surge_penalty
    return move_penalties[adj, patient_type]

@njit(cache=True)
def insert_patient(state, patient_type, to_block):
    """
    Returns the state that results from inserting a patient.

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.
      + `patient_type`: the type of the patient being inserted, either
           2: 'Stage 3-I', 1: 'Stage 3', or 0: 'Stage 2'
      + `to_block`: the block the patient inserted to

    Returns: a numpy array representing the state after the insert.
    """
    state[(patient_type * 16) + to_block] += 1


@njit(cache=True)
def move_patient(state, to_block, from_block):
    """
    Returns the state that results from moving a patient.

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.
      + `to_block`: the block the patient inserted to
      + `from_block`: the block the patient was removed from

    Returns: a numpy array representing the state after moving the
               patient.
    """
    patient_type = 0
    while patient_type < 2 and state[(patient_type * 16) + from_block] == 0:
        patient_type += 1
    state[(patient_type * 16) + to_block] += 1
    state[(patient_type * 16) + from_block] -= 1


@njit(cache=True)
def remove_patient(state, patient_type, from_block):
    """
    Returns the state that results from removing a patient.

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.
      + `patient_type`: the type of the patient being removed, either
           2: 'Stage 3-I', 1: 'Stage 3', or 0: 'Stage 2'
      + `from_block`: the block the patient was removed from

    Returns: a numpy array representing the state after removing the
               patient.
    """
    state[(patient_type * 16) + from_block] -= 1

@njit(cache=True)
def deteriorate_patient(state, patient_type, block):
    """
    Returns the state that results from an Gren patient deteriorating
    into an Amber patient.

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.
      + `patient_type`: the type of the patient deteriorating, either
           2: 'Stage 3-I', 1: 'Stage 3', or 0: 'Stage 2'
      + `block`: the block the deteriorating patient is

    Returns: a numpy array representing the state after the deterioration.
    """
    state[(patient_type * 16) + block] -= 1
    state[((patient_type + 1) * 16) + block] += 1

@njit(cache=True)
def get_available_insert_moves(state):
    """
    Lists all available places where a patient can be inserted.

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.

    Returns: a list of blocks that the patient can be inserted.    
    """
    occupancy = state[0:16] + state[16:32] + state[32:48]
    return (max_capacities > occupancy).nonzero()[0]


@njit(cache=True)
def get_available_actions(state, patient_type, actions_pool):
    """
    Lists all available actions that can happend when a patient of type
    `patient_type` arrives when the ward is in state `state`.
    An action takes the form:

    (b, d)

    where:
      - b is the block that the new patient will be inserted into
      - d is the block that patient will move to.

    In cases where no bed moved happen, we have (b = d).

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.
      + `patient_type`: the type of the patient to move, either
           2: 'Stage 3-I', 1: 'Stage 3', or 0: 'Stage 2'
      + `actions_pool`: a pre-assigned numpy empty array of
           size 16x17

    Returns: an array of actions, where each row is an integer abc, and the
             count of valid actions.
    """
    valid_count = 0

    isolation_full = (state[15] + state[31] + state[47]) == 2
    isolation_full_with_3i = state[(2 * 16) + 15] == 2

    # First, check if Stage 3-I can go to isolation unit
    if (patient_type == 2) and (not isolation_full):
        actions_pool[valid_count] = 1515
        valid_count += 1
        return actions_pool, valid_count

    available_blocks = get_available_insert_moves(state)

    # Second, check if Stage 3-I can displace someone from an isolation unit
    if (patient_type == 2) and isolation_full and (not isolation_full_with_3i):
        for to_block in available_blocks:
            actions_pool[valid_count] = 1500 + to_block
            valid_count += 1
        if state[15] > 0:
            actions_pool[valid_count] = 1516
            valid_count += 1
        return actions_pool, valid_count

    # Case A: Direct Insert (to_block == insert_block)
    for insert_block in available_blocks:
        actions_pool[valid_count] = (100 * insert_block) + insert_block
        valid_count += 1
    # Case B: Bed Move (to_block != insert_block)
    for insert_block in range(16):
        if state[(patient_type * 16) + insert_block] < max_capacities[insert_block]:
            col_sum = state[insert_block] + state[16 + insert_block] + state[32 + insert_block] 
            if col_sum > 0:
                for to_block in available_blocks:
                    if insert_block != to_block:
                        if not (isolation_full_with_3i and (insert_block == 15)):
                            if not (insert_block == 15 and patient_type == 0 and state[15] > 0):
                                actions_pool[valid_count] = (100 * insert_block) + to_block
                                valid_count += 1
                if (state[insert_block] > 0) and (patient_type != 0):
                    actions_pool[valid_count] = (100 * insert_block) + 16
                    valid_count += 1
    return actions_pool, valid_count



@njit(cache=True)
def find_idx_of_patient_to_move(
    block,
    patient_type,
    patients_blocks,
    patients_types
):
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
        if (patients_types[i] == patient_type) & (block == patients_blocks[i]):
            return i
    return -1
