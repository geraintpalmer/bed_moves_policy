import numpy as np
import ward
import random
from numba import njit

@njit(cache=True)
def choose_random_action(actions_pool, valid_count):
    """
    Chooses an action randomly from a list of blocks.

    Arguments:
      + `actions_pool`: a pre-assigned numpy empty array of
           size 9 + (9 * 2 * 8)
      + `valid_count`: the number of actions that are valid

    Returns: an action.
    """
    idx = np.random.randint(0, valid_count)
    return actions_pool[idx]


@njit(cache=True)
def choose_best_action(
    state,
    hash_state_only,
    equivalence_idx,
    patient_type,
    default_future_reward,
    actions_pool,
    valid_count,
    Q_index_map,
    qval_array
):
    """
    Chooses the best action.

    Arguments:
      + `state` a numpy array of 27 integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient arriving, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `actions_pool`: a pre-assigned numpy empty array of
           size 9 + (9 * 2 * 8)
      + `valid_count`: the number of actions that are valid
      + `Q_index_map`: dictionary of stateaction to indices
      + `qval_array`: array of q-values

    Returns: an action, and the Q-value associated with that
             state-best-action pair
    """
    best_Q = -np.float32(np.inf)
    best_noisy_Q = -np.float32(np.inf)
    best_idx = -1
    for i in range(valid_count):
        a = np.int64(ward.inverse_action(actions_pool[i], equivalence_idx))
        key = hash_state_only + a
        if key in Q_index_map:
            idx = Q_index_map[key]
            Q = qval_array[np.int64(idx)]
        else:
            Q = np.float32(default_future_reward)
        noise = np.float32(np.random.random() * np.float32(1e-5))
        if best_noisy_Q < (Q + noise):
            best_Q = Q
            best_noisy_Q = Q + noise
            best_idx = i
    return actions_pool[best_idx], best_Q


@njit(cache=True)
def choose_action(
    state,
    patient_type,
    epsilon,
    default_future_reward,
    Q_index_map,
    qval_array,
    actions_pool,
    buffer_state
):
    """
    Randomly chooses an action (1-epsilon) of the time.
    Otherwise chooses the best.

    Arguments:
      + `state` a numpy array of 27 integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient arriving, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `epsilon`: a probability, float between 0 and 1
           (low: explore more, high: exploit more)
      + `Q_index_map`: dictionary of stateaction to indices
      + `qval_array`: array of q-values
      + `actions_pool`: a pre-assigned numpy empty array of
           size 9 + (9 * 2 * 8)

    Returns: a tuple of two things: the best action (None if no action
               can be taken), the q-value associated with that best action
               (only if choosing the best action, None otherwise)
    """
    actions_pool, valid_count = ward.get_available_actions(
        state=state,
        patient_type=patient_type,
        actions_pool=actions_pool
    )
    hash_state_only, equivalence_idx = ward.get_representative_hash_state(
        state=state,
        patient_type=patient_type,
        buffer_state=buffer_state
    )

    if np.random.random() < epsilon:
        a, Qa = choose_best_action(
            state=state,
            hash_state_only=hash_state_only,
            equivalence_idx=equivalence_idx,
            patient_type=patient_type,
            default_future_reward=default_future_reward,
            actions_pool=actions_pool,
            valid_count=valid_count,
            Q_index_map=Q_index_map,
            qval_array=qval_array
        )
        next_hash_state = hash_state_only + ward.inverse_action(a, equivalence_idx)
        return a, Qa, next_hash_state, equivalence_idx

    a = choose_random_action(
        actions_pool=actions_pool,
        valid_count=valid_count
    )
    next_hash_state = hash_state_only + ward.inverse_action(a, equivalence_idx)
    return a, None, next_hash_state, equivalence_idx


@njit(cache=True)
def exploit_policy(state, patient_type, policy, actions_pool, buffer_state):
    """
    Choose an action by exploiting the policy.

    Arguments:
      + `state` a numpy array of 27 integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the arriving patient (0, 1, or 2)
      + `policy`: the Numba typed dictionary mapping hash states to best actions
      + `actions_pool`: a pre-assigned numpy empty array of
           size 9 + (9 * 2 * 8)

    Returns: the best action.
    """
    hash_state_only, equivalence_idx = ward.get_representative_hash_state(
        state=state,
        patient_type=patient_type,
        buffer_state=buffer_state
    )

    if hash_state_only in policy:
        a = policy[hash_state_only]
        return ward.permute_action(a, equivalence_idx)

    actions_pool, valid_count = ward.get_available_actions(
        state=state,
        patient_type=patient_type,
        actions_pool=actions_pool
    )
    return choose_random_action(
        actions_pool=actions_pool,
        valid_count=valid_count
    )
