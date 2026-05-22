import numpy as np
from numba import njit
import itertools

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

tiling_4 = np.array([0, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2])
tiling_3 = np.array([0, 1, 1, 1, 1, 2, 1, 2])

max_possible_hash = np.iinfo(np.int64).max

T1 = np.array([3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64)
T2 = np.array([0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64)
T3 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15], dtype=np.int64)
T4 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 13, 12, 15], dtype=np.int64)
T5 = np.array([4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64)
transforms = [T5, T4, T3, T2, T1]

def generate_equivalence_permutations(transforms):
    """
    Generates the list of all 32 permutations out of the five
    transforms, such that all those that use the swap transform
    are in the second half of the list.

    Arguments:
      - `transforms`: the list of 5 transforms.

    Returns: a list of all 32 possible combinations of transforms,
      applied to the full 48 arrays.
    """
    n_transforms = len(transforms)
    equivalence_permutations = np.zeros((2**n_transforms, 48), dtype=np.int64)
    for j, vertex in enumerate(itertools.product([0, 1], repeat=n_transforms)):
        original = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64)
        for i, v in enumerate(vertex):
            if v == 1:
                original = original[transforms[i]]
        equivalence_permutations[j,:16] = original
        equivalence_permutations[j,16:32] = original + 16
        equivalence_permutations[j,32:] = original + 32
    return equivalence_permutations


equivalence_permutations = generate_equivalence_permutations(transforms)


@njit(cache=True)
def get_hash_state_only(state, patient_type):
    """
    Returns a hashable version of the state - not including the action.

    Arguments:
      + `state`: a numpy array representing the state of the system,
      + `patient_type`: an integer representing the arriving customer
           type.

    Returns: an integer representation of the state, with 0 placeholder
    for an action.
    """
    return (100000 * (hash_weights * state).sum()) + (patient_type * 10000)


@njit(cache=True)
def get_representative_hash_state(state, patient_type, buffer_state):
    """
    Returns a hashable version of the state - not including the action.
    This returns the representative hash - it will return the _same_ hash
    for each state in the same equivalence class.

    Arguments:
      + `state`: a numpy array representing the state of the system,
      + `patient_type`: an integer representing the arriving customer
           type.
      + `buffer_state`: a 48-array with pre-allocated memory. 

    Returns: an integer representation of the state, with 0 placeholder
    for an action, and the index of the original state in the
    equivalence_permutations array.
    """
    current_hash = max_possible_hash
    current_idx = 0
    for equivalence_idx in range(32):
        p = equivalence_permutations[equivalence_idx]
        for i in range(48):
            buffer_state[i] = state[p[i]]
        h = get_hash_state_only(state=buffer_state, patient_type=patient_type)
        if h <  current_hash:
            current_hash = h
            current_idx = equivalence_idx
    return current_hash, current_idx


@njit(cache=True)
def dehash_state(hash_state):
    """
    Returns the matrix representation and patient type,
    from the hashed state integer.

    Arguments:
      - `hash_state`: the integer representation of the state only

    Returns: a tuple of:
      - `state`: the matrix representation of the state
      - `patient_type`: an integer in {0, 1, 2} representing the patient type.
    """
    patient_type = (hash_state % 100000) // 10000
    remainder = hash_state // 100000
    state = np.zeros(48, dtype=np.int16)

    order = np.argsort(hash_weights)[::-1]
    
    for i in order:
        weight = hash_weights[i]
        state[i] = remainder // weight
        remainder %= weight
            
    return state, patient_type


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
    a1 = action_hash // 100
    return a1, a2


@njit(cache=True)
def get_hash_stateaction(state, patient_type, action, buffer_state):
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
    hash_state_only, idx = get_representative_hash_state(
        state=state,
        patient_type=patient_type,
        buffer_state=buffer_state
    )
    return hash_state_only + action, idx


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
def inverse_action(a, equivalence_idx):
    """
    Transforms the representative actions a1 and a2 into their
    original actions, if the permutation used to get the
    representative actions was `equivalence_idx`.

    Arguments:
      - `a`: the 4 digit action hash
      - `equivalence_idx`: the permutation used to go from the
      current state to the representative state.

    Returns: the transformed a1 and a2.
    """
    a1, a2 = dehash_action(a)
    if equivalence_idx < 16:
        a1 = equivalence_permutations[equivalence_idx, a1]
        if a2 < 16:
            a2 = equivalence_permutations[equivalence_idx, a2]
    else:
        a1 = T5[a1]
        a1 = equivalence_permutations[equivalence_idx, a1]
        a1 = T5[a1]
        if a2 < 16:
            a2 = T5[a2]
            a2 = equivalence_permutations[equivalence_idx, a2]
            a2 = T5[a2]
    return (a1 * 100) + a2


@njit(cache=True)
def get_stage2_staffing(state_row):
    """
    Gets the staffing requirements for Stage 2 patients only.
    Converts the 4, 4, 4, and 3 bed ward configurations into integers,
    and looks up the staffing tiling resource requirement

    Arguments:
      + `state`: a numpy array representing the state of the system

    Returns: an integer number of staff required.
    """
    staffing = 0
    idx41 = 0
    idx42 = 0
    idx43 = 0
    for i in range(4):
        idx41 += state_row[3 - i] << i
        idx42 += state_row[7 - i] << i
        idx43 += state_row[11 - i] << i
    idx3 = 0
    for i in range(3):
        idx3 += state_row[14 - i] << i
    staffing += tiling_4[idx41] + tiling_4[idx42] + tiling_4[idx43]
    staffing += tiling_3[idx3]
    return staffing


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
def move_patient(state, patient_type, to_block, from_block):
    """
    Returns the state that results from moving a patient.

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.
      + `patient_type`: the type of the patient being inserted, either
           2: 'Stage 3-I', 1: 'Stage 3', or 0: 'Stage 2'
      + `to_block`: the block the patient inserted to
      + `from_block`: the block the patient was removed from

    Returns: a numpy array representing the state after moving the
               patient.
    """
    find_patient_type_to_move(state=state, from_block=from_block)
    state[(patient_type * 16) + from_block] -= 1
    if to_block != 16:
        state[(patient_type * 16) + to_block] += 1


@njit(cache=True)
def find_patient_type_to_move(state, from_block):
    """
    When moving a patient from block `from_block`,
    returns the type of patient that is to be moved.

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.
      + `from_block`: the block the patient was removed from

    Returns: an integer {0, 1, 2} representing the type of patient
        to move.
    """
    patient_type = 0
    while patient_type < 2 and state[(patient_type * 16) + from_block] == 0:
        patient_type += 1
    return patient_type


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
    Returns the state that results from a patient deteriorating.

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
def improve_patient(state, patient_type, block):
    """
    Returns the state that results from a patient improving.

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.
      + `patient_type`: the type of the patient improving, either
           2: 'Stage 3-I', 1: 'Stage 3', or 0: 'Stage 2'
      + `block`: the block the improving patient is

    Returns: a numpy array representing the state after the improvement.
    """
    state[(patient_type * 16) + block] -= 1
    state[((patient_type - 1) * 16) + block] += 1


@njit(cache=True)
def get_available_noniso_insert_moves(state):
    """
    Lists all available places where a patient can be inserted.

    Arguments:
      + `state` an array of 48 integers {0, 1, 2} representing
           the state of the ward.

    Returns: a list of blocks that the patient can be inserted.    
    """
    occupancy = state[0:15] + state[16:31] + state[32:47]
    return (occupancy < 1).nonzero()[0]


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

    isolation_has_0 = state[15] > 0
    isolation_full = (state[15] + state[31] + state[47]) == 2
    isolation_full_with_3i = state[(2 * 16) + 15] == 2
    available_blocks = get_available_noniso_insert_moves(state)

    if patient_type == 2:
        if isolation_full_with_3i:
            for a1 in available_blocks:
                actions_pool[valid_count] = a1 * 101
                valid_count += 1
            beds_with_0 = np.where(state[:15] > 0)[0]
            for a1 in beds_with_0:
                for a2 in available_blocks:
                    actions_pool[valid_count] = (a1 * 100) + a2
                    valid_count += 1
                actions_pool[valid_count] = (a1 * 100) + 16
                valid_count += 1
            beds_with_1 = np.where(state[16:31] > 0)[0]
            for a1 in beds_with_1:
                for a2 in available_blocks:
                    actions_pool[valid_count] = (a1 * 100) + a2
                    valid_count += 1
        elif isolation_full:
            for a2 in available_blocks:
                actions_pool[valid_count] = 1500 + a2
                valid_count += 1
            if isolation_has_0:
                actions_pool[valid_count] = 1516
                valid_count += 1
        elif not isolation_full:
            actions_pool[valid_count] = 1515
            valid_count += 1
    if patient_type == 1:
        for a1 in available_blocks:
            actions_pool[valid_count] = a1 * 101
            valid_count += 1
        if not isolation_full:
            actions_pool[valid_count] = 1515
            valid_count += 1
        beds_with_0 = np.where(state[:15] > 0)[0]
        for a1 in beds_with_0:
            for a2 in available_blocks:
                if a1 != a2:
                    actions_pool[valid_count] = (a1 * 100) + a2
                    valid_count += 1
            actions_pool[valid_count] = (a1 * 100) + 16
            valid_count += 1
        beds_with_2 = np.where(state[32:47] > 0)[0]
        for a1 in beds_with_2:
            for a2 in available_blocks:
                if a1 != a2:
                    actions_pool[valid_count] = (a1 * 100) + a2
                    valid_count += 1
        if isolation_has_0:
            for a2 in available_blocks:
                actions_pool[valid_count] = 1500 + a2
                valid_count += 1
            actions_pool[valid_count] = 1516
            valid_count += 1
    if patient_type == 0:
        for a1 in available_blocks:
            actions_pool[valid_count] = a1 * 101
            valid_count += 1
        if (len(available_blocks) == 0) and (not isolation_full):
            actions_pool[valid_count] = 1515
            valid_count += 1
        beds_with_1 = np.where(state[16:31] > 0)[0]
        for a1 in beds_with_1:
            for a2 in available_blocks:
                if a1 != a2:
                    actions_pool[valid_count] = (a1 * 100) + a2
                    valid_count += 1
        beds_with_2 = np.where(state[32:47] > 0)[0]
        for a1 in beds_with_2:
            for a2 in available_blocks:
                if a1 != a2:
                    actions_pool[valid_count] = (a1 * 100) + a2
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
