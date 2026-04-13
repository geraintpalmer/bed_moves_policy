import numpy as np
import ward
from math import exp
from numba import njit

@njit(cache=True)
def merge_sorted_qvals(keys1, vals1, hits1, keys2, vals2, hits2):
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
    vals_n = np.empty(unique_count, dtype=np.float64)
    hits_n = np.empty(unique_count, dtype=np.int64)

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
def get_best_future_reward(state, patient_type, Qvals, just_chose_best, prev_best_Q):
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

    Returns: the maximum expected future reward from following the
      best actions from this state onwards.
    """
    if just_chose_best:
        return prev_best_Q

    available_as = ward.get_available_insert_moves(state=state)
    hash_state_only = ward.get_hash_state_only(
        state=state,
        patient_type=patient_type,
        hash_weights=ward.hash_weights
    )

    best_Q = -np.inf
    for a in available_as:
        hash_state = hash_state_only + a
        if hash_state in Qvals:
            Q = Qvals[hash_state]
            if Q > best_Q:
                best_Q = Q

    return best_Q


@njit(cache=True, fastmath=True)
def update_Q_values(
    hash_state,
    next_state,
    next_patient_type,
    next_action,
    Qvals,
    hits,
    reward,
    learning_rate,
    discount_factor,
    just_chose_best,
    prev_best_Q,
    default_future_reward
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

    Returns: (updates the Qvals and hits dictionaries) and returns
             the hash state of the newly reached state.
    """
    best_future_reward = get_best_future_reward(
        state=next_state,
        patient_type=next_patient_type,
        Qvals=Qvals,
        just_chose_best=just_chose_best,
        prev_best_Q=prev_best_Q
    )
    if np.isinf(best_future_reward):
        best_future_reward = default_future_reward / (1 - discount_factor)

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

    next_hash_state = ward.get_hash_stateaction(
        state=next_state,
        patient_type=next_patient_type,
        action=next_action,
        hash_weights=ward.hash_weights
    )
    return next_hash_state


@njit(cache=True, fastmath=True)
def transform_cost(cost):
    """
    Transforms the cost (C) into a reward (R) via: R = -C

    Arguments:
      + `cost` the cost since the last timestamp

    Returns: a reward.
    """
    return -cost


@njit(cache=True)
def initialise_qvals(keys_array, qval_array, Qvals, hits):
    """
    Initialises Qvals dictionary with some previously learned
    Q-values.

    Arguments:
      + `keys_array`: a numpy array containing the hashed stateaction pairs
      + `qval_array`: a numpy array containing the learned q-values
      + `Qvals`: an empty typed dictionary for the Q-values
      + `hits`: an empty typed dictionary for the hits.
    """
    for k, v in zip(
        keys_array,
        qval_array
    ):
        Qvals[k] = v
        hits[k] = np.int64(0)


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
    running_max = 0.0
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
def get_arrays_from_dicts(Qvals, hits):
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
