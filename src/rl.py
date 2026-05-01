import numpy as np
import ward
from math import exp
from numba import njit

@njit(cache=True)
def get_unique_tails(keys1, vals1, hits1, keys2, vals2, hits2):
    """
    Merges two sorted arrays of keys, vals, and hits.

    Arguments
      - `keys1`: a sorted numpy array of int64, the state-action
           pair hashes
      - `vals1`: a sorted numpy array of float64, the Q-values
           associated with the state-action pairs
      - `hits1`: a sorted numpy array of int64, the number of
           hits per state-action pair
      - `keys2`: a sorted numpy array of int64, the state-action
           pair hashes
      - `vals2`: a sorted numpy array of float64, the Q-values
           associated with the state-action pairs
      - `hits2`: a sorted numpy array of int64, the number of
           hits per state-action pair
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
    Updates the master arrays of keys, vals, and hits with the new updates.

    Arguments
      - `vals1`: a sorted numpy array of float64, the Q-values
           associated with the state-action pairs
      - `hits1`: a sorted numpy array of int64, the number of
           hits per state-action pair
      - `vals2`: a sorted numpy array of float64, the Q-values
           associated with the state-action pairs
      - `hits2`: a sorted numpy array of int64, the number of
           hits per state-action pair
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
      + `Qvals`: a dictionary of stateaction to q-values
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
        patient_type=patient_type,
        hash_weights=ward.hash_weights
    )

    best_Q = -np.float32(np.inf)
    for i in range(valid_count):
        hash_state = hash_state_only + actions_pool[i]
        if hash_state in Q_index_map:
            idx = Q_index_map[hash_state]
            Q = qval_array[np.int64(idx)]
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
      + `state`: a numpy array representing the state the
           system has just reached
      + `patient_type`: an integer representing the arriving
           customer type
      + `action`: the action that has been chosen
      + `Qvals`: a dictionary of stateaction to q-values
      + `hits`: a dictionary of stateaction to hits
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
    if np.isinf(best_future_reward):
        best_future_reward = default_future_reward / (np.float32(1.0) - discount_factor)

    next_hash_state = ward.get_hash_stateaction(
        state=next_state,
        patient_type=next_patient_type,
        action=next_action,
        hash_weights=ward.hash_weights
    )

    if hash_state in Q_index_map:
        idx = np.int64(Q_index_map[hash_state])
        oldQ = qval_array[idx]
        h = hits_array[idx] + np.int16(1)
    else:
        if max_idx >= len(qval_array): # skip learning, no space left for new state-action pairs
            return next_hash_state, max_idx

        idx = np.int64(max_idx)
        Q_index_map[hash_state] = np.int32(max_idx)
        max_idx += np.int32(1)
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
    qval_array[idx] = np.float32(newQ)
    hits_array[idx] = np.int16(h)
    return next_hash_state, max_idx



@njit(cache=True)
def initialise_qvals(initial_states_array, initial_qval_array, states_array, qval_array, hits_array, Q_index_map):
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
def initialise_policy(keys_array, qval_array, policy):
    """
    Initialises policy dictionary with the previously
    learned Q-values.

    Arguments:
      + `keys_array`: a numpy array containing the hashed stateaction pairs
      + `qval_array`: a numpy array containing the learned q-values
      + `policy`: an empty typed dictionary for the policy.
    """
    running_max = np.float32(0.0)
    for k, v in zip(keys_array, qval_array):
        hash_state_only, a = ward.get_state_action_from_hashstate(k)
        if hash_state_only in policy:
            if running_max < v:
                policy[hash_state_only] = a
                running_max = v
        else:
            policy[hash_state_only] = a
            running_max = v


@njit(cache=True)
def block_sort_arrays(states_array, qval_array, hits_array, m, max_idx):
    """
    Gets numpy arrays from the Numba typed dictionary.

    Arguments:
      + `Qvals`: a typed dictionary mapping hash states to Q-values
      + `hits`: a typed dictionary mapping hash states to the number of hits.

    Returns:
      + `n` the number of hash states discovered so far
      + `keys_arr` the numpy array of hash states
      + `q_arr` the numpy array of q-values
      + `hits_arr` the numpy array of numbers of hits
    """
    max_idx = np.int64(max_idx)
    states_array = states_array[:max_idx]
    qval_array = qval_array[:max_idx]
    hits_array = hits_array[:max_idx]

    newfound_states = states_array[m:]
    idx_order = np.argsort(newfound_states)

    states_array[m:] = states_array[m:][idx_order]
    qval_array[m:] = qval_array[m:][idx_order]
    hits_array[m:] = hits_array[m:][idx_order]

    return max_idx, states_array, qval_array, hits_array
