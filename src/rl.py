import numpy as np
import ward
from math import exp
from numba import njit, typed, types

worst_Q = np.finfo(np.float32).min
check_worst = 0.99999 * np.finfo(np.float32).min

@njit(cache=True)
def get_unique_tails(keys1, vals1, hits1, keys2, vals2, hits2):
    """
    Merges the tails of two sorted sets of keys, vals, and hits arrays.

    Arguments
      - `keys1`: a sorted numpy array of int64, the state-action
           pair hashes (tail of trial 1)
      - `vals1`: a sorted numpy array of float32, the Q-values
           associated with the state-action pairs (tail of trial 1)
      - `hits1`: a sorted numpy array of int16, the number of
           hits per state-action pair (tail of trial 1)
      - `keys2`: a sorted numpy array of int64, the state-action
           pair hashes (tail of trial 2)
      - `vals2`: a sorted numpy array of float32, the Q-values
           associated with the state-action pairs (tail of trial 2)
      - `hits2`: a sorted numpy array of int16, the number of
           hits per state-action pair (tail of trial 2)
    Returns: the same three arrays merge-sorted.
    """
    idx_1 = 0
    idx_2 = 0
    unique_count = 0
    while idx_1 < len(keys1) and idx_2 < len(keys2):
        if keys1[idx_1] == keys2[idx_2]:
            idx_1 += 1
            idx_2 += 1
        elif keys1[idx_1] < keys2[idx_2]:
            idx_1 += 1
        else:
            idx_2 += 1
        unique_count += 1
    unique_count += (len(keys1) - idx_1) + (len(keys2) - idx_2)

    keys_n = np.empty(unique_count, dtype=np.int64)
    vals_n = np.empty(unique_count, dtype=np.float32)
    hits_n = np.empty(unique_count, dtype=np.int16)

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
            keys_n[idx_n] = keys1[idx_1]
            hits_n[idx_n] = sum_hits
            if sum_hits == 0:
                vals_n[idx_n] = vals1[idx_1]
            else:
                vals_n[idx_n] = ((vals1[idx_1] * hits1[idx_1]) + (vals2[idx_2] * hits2[idx_2])) / sum_hits
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

    return keys_n, vals_n, hits_n

@njit(cache=True)
def update_master_head_inplace(vals1, hits1, vals2, hits2):
    """
    Updates the heads of the master arrays of keys, vals,
    and hits with the new updates.

    Arguments
      - `vals1`: a head of a numpy array of float32, the Q-values
           associated with the state-action pairs (master array)
      - `hits1`: a head of numpy array of int16, the number of
           hits per state-action pair (master array)
      - `vals2`: a head of numpy array of float32, the Q-values
           associated with the state-action pairs (new array)
      - `hits2`: a sorted numpy array of int16, the number of
           hits per state-action pair (new array)
    """
    for i in range(len(hits1)):
        sum_hits = hits1[i] + hits2[i]
        if sum_hits == 0:
            vals1[i] = vals1[i]
        else:
            vals1[i] = ((vals1[i] * hits1[i]) + (vals2[i] * hits2[i])) / sum_hits
        hits1[i] = sum_hits


@njit(cache=True)
def get_best_future_reward(
    state,
    patient_type,
    qval_array,
    Q_index_map,
    just_chose_best,
    prev_best_Q,
    actions_pool
):
    """
    Returns the maximum future reward if taking the optimal action
    when in the future state.

    Arguments:
      + `state`: a numpy array representing the state the
          system has just reached
      + `patient_type`: an integer representing the arriving
          customer type
      + `Q_index_map`: dictionary of stateaction to indices
      + `qval_array`: array of of q-values
      + `just_chose_best`: a Boolean representing if the
             simulation chose the best action in the previous step
      + `prev_best_Q`: the previously chosen best q-value
      + `actions_pool`: a pre-assigned numpy empty array of
           size 9 + (9 * 2 * 8)

    Returns: the maximum expected future reward from following the
      best actions from this state onwards.
    """
    if just_chose_best:
        return prev_best_Q

    actions_pool, valid_count = ward.get_available_actions(
        state=state,
        patient_type=patient_type,
        actions_pool=actions_pool
    )
    hash_state_only = ward.get_hash_state_only(
        state=state,
        patient_type=patient_type
    )

    best_Q = worst_Q
    for i in range(valid_count):
        hash_state = hash_state_only + np.int64(actions_pool[i])
        if hash_state in Q_index_map:
            idx = Q_index_map[hash_state]
            Q = qval_array[idx]
            if Q > best_Q:
                best_Q = Q

    return best_Q


@njit(cache=True, fastmath=True)
def update_Q_values(
    hash_state,
    next_state,
    next_patient_type,
    next_action,
    states_array,
    qval_array,
    hits_array,
    Q_index_map,
    max_idx,
    reward,
    learning_rate,
    discount_factor,
    just_chose_best,
    prev_best_Q,
    default_future_reward,
    actions_pool
):
    """
    Updates the Q-values according to the Q-learning update:

    Arguments:
      + `hash_state`: the hash state to update
      + `next_state`: a numpy array representing the state the
           system has just reached
      + `next_patient_type`: an integer representing the arriving
           customer type
      + `next_action`: the action that has been chosen
      + `states_array`: array of states
      + `qval_array`: array of of q-values
      + `hits_array`: array of hits
      + `Q_index_map`: dictionary of stateaction to indices
      + `max_idx`: the number of state-action pairs for which a q-value
           has been found, the next index to place any newly discovered
           state-action.
      + `reward`: the reward obtained by reaching the next state
      + `learning_rate`: the learning rate of the Q-learning
           algorithm (a number between 0 and 1)
      + `discount_factor`: the discount factor of the Q-learning
           algorithm (a number between 0 and 1)
      + `just_chose_best`: a Boolean representing if the
             simulation chose the best action in the previous step
      + `prev_best_Q`: the previously chosen best q-value
      + `default_future_reward`: the future reward given if all
           future actions unexplored
      + `actions_pool`: a pre-assigned numpy empty array of
           size 9 + (9 * 2 * 8)

    Returns: (updates the Qvals and hits dictionaries) and returns
             the hash state of the newly reached state.
    """
    best_future_reward = get_best_future_reward(
        state=next_state,
        patient_type=next_patient_type,
        qval_array=qval_array,
        Q_index_map=Q_index_map,
        just_chose_best=just_chose_best,
        prev_best_Q=prev_best_Q,
        actions_pool=actions_pool
    )
    if best_future_reward < check_worst:
        best_future_reward = default_future_reward / (np.float32(1.0) - discount_factor)

    next_hash_state = ward.get_hash_stateaction(
        state=next_state,
        patient_type=next_patient_type,
        action=next_action
    )

    try:
        idx = np.int64(Q_index_map[hash_state])
        oldQ = qval_array[idx]
        h = hits_array[idx] + np.int16(1)
    except:
        if max_idx >= len(qval_array): # skip learning, no space left for new state-action pairs
            return next_hash_state, max_idx

        idx = max_idx
        Q_index_map[hash_state] = np.int32(max_idx)
        max_idx += 1
        oldQ = np.float32(0.0)
        h = np.int16(1)

    newQ = (
        ((1.0 - learning_rate) * oldQ)
        + (learning_rate * (
            reward + (
                discount_factor * best_future_reward
            )
        ))
    )
    states_array[idx] = hash_state
    qval_array[idx] = np.float32(newQ)
    hits_array[idx] = np.int16(h)
    return next_hash_state, max_idx



@njit(cache=True)
def initialise_qvals(
    initial_states_array,
    initial_qval_array,
    states_array,
    qval_array,
    hits_array,
    Q_index_map
):
    """
    Initialises Q-values data structure.

    Arguments:
      + `initial_states_array`: a numpy array containing the initial hashed stateaction pairs
      + `initial_qval_array`: a numpy array containing the initial qvalues
      + `states_array`: an empty numpy array for the hashed stateaction pairs
      + `qval_array`: an empty numpy array for the learned q-values
      + `hits_array`: an empty numpy array for the hits
      + `Q_index_map`: an empty typed dictionary for mapping states to indices
    """
    for i in range(len(initial_states_array)):
        s = initial_states_array[i]
        states_array[i] = s
        qval_array[i] = initial_qval_array[i]
        hits_array[i] = np.int16(0)
        Q_index_map[s] = np.int32(i)

@njit(cache=True)
def initialise_policy_dict(keys_array, policy_array, policy):
    """
    Initialises policy dictionary with the previously
    learned Q-values.

    Arguments:
      + `keys_array`: a numpy array containing the hashed states
      + `policy_array`: a numpy array containing the learned q-values
      + `policy`: an empty typed dictionary for the policy.
    """
    for i in range(len(keys_array)):
        k = keys_array[i]
        a = policy_array[i]
        policy[k] = np.int32(a)

@njit(cache=True)
def initialise_policy(keys_array, qval_array):
    """
    Initialises policy dictionary with the previously
    learned Q-values.

    Arguments:
      + `keys_array`: a numpy array containing the hashed stateaction pairs
      + `qval_array`: a numpy array containing the learned q-values
    """
    best_indices = typed.Dict.empty(key_type=types.int64, value_type=types.int32)

    for idx in range(len(keys_array)):
        k = keys_array[idx]
        qval = qval_array[idx]
        hash_state_only, a = ward.get_state_action_from_hashstate(k)

        if hash_state_only in best_indices:
            prev_best_idx = best_indices[hash_state_only]
            prev_best_Q = qval_array[prev_best_idx] 
            if qval > prev_best_Q:
                best_indices[hash_state_only] = np.int32(idx)
        else:
            best_indices[hash_state_only] = np.int32(idx)

    j = len(best_indices)
    out_keys_array = np.empty(j, dtype=np.int64)
    out_policy_array = np.empty(j, dtype=np.int16)

    for i, (k, idx) in enumerate(best_indices.items()):
        out_keys_array[i] = k
        stateaction = keys_array[idx]
        hash_state_only, a = ward.get_state_action_from_hashstate(stateaction)
        out_policy_array[i] = np.int16(a)

    return out_keys_array, out_policy_array



@njit(cache=True)
def block_sort_arrays(states_array, qval_array, hits_array, m, max_idx):
    """
    Sorts the tails of the states, qval, and hits arrays.

    Arguments:
      + `states_array`: a numpy array containing the hashed stateaction pairs
      + `qval_array`: a numpy array containing the learned q-values
      + `hits_array`: a numpy array containing the number of hits for each stateaction pair
      + `m`: the number of previously-learned stateactions, no need to sort these
      + `max_idx`: the number of stateactions.

    Returns:
      + `n` the number of hash states discovered so far
      + `states_array` the numpy array of hash states
      + `qval_array` the numpy array of q-values
      + `hits_array` the numpy array of numbers of hits
    """
    states_array = states_array[:max_idx]
    qval_array = qval_array[:max_idx]
    hits_array = hits_array[:max_idx]

    newfound_states = states_array[m:]
    idx_order = np.argsort(newfound_states)

    states_array[m:] = states_array[m:][idx_order]
    qval_array[m:] = qval_array[m:][idx_order]
    hits_array[m:] = hits_array[m:][idx_order]

    return max_idx, states_array, qval_array, hits_array
