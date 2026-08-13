import numpy as np
import ward
import random
from numba import njit

EPSILON_GREEDY = 0
MIXTURE = 1

tau = 0.3
top3_probs = np.exp(tau * np.array([-1, -2, -3])) / np.exp(tau * np.array([-1, -2, -3])).sum()

@njit(cache=True)
def choose_random_action(actions_pool, valid_count):
    """
    Chooses an action randomly from a list of blocks.

    Arguments:
      + `actions_pool`: a pre-assigned numpy empty array of
           size 15 * 16
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
      + `state` a numpy array of 45 integers {0, 1, 2} representing
           the state of the ward.
      + `patient_type`: the type of the patient arriving, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `actions_pool`: a pre-assigned numpy empty array of
           size 15 * 16
      + `valid_count`: the number of actions that are valid
      + `Q_index_map`: dictionary of stateaction to indices
      + `qval_array`: array of q-values

    Returns: an action, and the Q-value associated with that
             state-best-action pair
    """
    bestQ = -np.float32(np.inf)
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
            bestQ = Q
            best_noisy_Q = Q + noise
            best_idx = i
    return actions_pool[best_idx], bestQ


@njit(cache=True)
def choose_action(
    state,
    patient_type,
    selection_policy,
    epsilon,
    default_future_reward,
    Q_index_map,
    qval_array,
    actions_pool,
    buffer_state,
    q_value_pool,
    fixed_mask
):
    """
    Randomly chooses an action (1-epsilon) of the time.
    Otherwise chooses the best.

    Arguments:
      + `state` a numpy array of 45 integers {0, 1, 2} representing
           the state of the ward.
      + `patient_type`: the type of the patient arriving, either
           2: 'red', 1: 'amber', or 0: 'green'
      + selection_policy: an integer representing the selection policy
          to use (EPSILON_GREEDY or SOFTMAX)
      + `epsilon`: a probability for epsilon-greedy: float between
           0 and 1 (low: explore more, high: exploit more)
      + `Q_index_map`: dictionary of stateaction to indices
      + `qval_array`: array of q-values
      + `actions_pool`: a pre-assigned numpy empty array of
           size 15 * 16

    Returns: a tuple of two things: the best action (None if no action
               can be taken), the q-value associated with that best action
               (only if choosing the best action, None otherwise)
    """
    ward.fixed_point_decision_tree(
        state=state,
        not_composed_of=ward.not_composed_of,
        fixed_mask=fixed_mask
    )
    actions_pool, valid_count = ward.get_available_actions(
        state=state,
        patient_type=patient_type,
        actions_pool=actions_pool,
        fixed_mask=fixed_mask
    )

    hash_state_only, equivalence_idx = ward.get_representative_hash_state(
        state=state,
        patient_type=patient_type,
        buffer_state=buffer_state
    )

    if selection_policy == EPSILON_GREEDY:
        return epsilon_greedy(
            state=state,
            hash_state_only=hash_state_only,
            equivalence_idx=equivalence_idx,
            valid_count=valid_count,
            patient_type=patient_type,
            epsilon=epsilon,
            default_future_reward=default_future_reward,
            Q_index_map=Q_index_map,
            qval_array=qval_array,
            actions_pool=actions_pool
        )
    elif selection_policy == MIXTURE:
        return top3_mixture(
            state=state,
            hash_state_only=hash_state_only,
            equivalence_idx=equivalence_idx,
            valid_count=valid_count,
            patient_type=patient_type,
            epsilon=epsilon,
            default_future_reward=default_future_reward,
            Q_index_map=Q_index_map,
            qval_array=qval_array,
            actions_pool=actions_pool,
            q_value_pool=q_value_pool
        )


@njit(cache=True)
def epsilon_greedy(
    state,
    hash_state_only,
    equivalence_idx,
    valid_count,
    patient_type,
    epsilon,
    default_future_reward,
    Q_index_map,
    qval_array,
    actions_pool
):
    """
    Chooses an action using the epsilon-greedy policy.
    Randomly chooses an action `epsilon` of the time.
    Otherwise chooses the best.

    Arguments:
      + `state` a numpy array of 45 integers {0, 1, 2} representing
           the state of the ward.
      + `hash_state_only`::::
      + `equivalence_idx`::::
      + `valid_count`::::
      + `patient_type`: the type of the patient arriving, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `epsilon`: a probability for epsilon-greedy: float between
           0 and 1 (low: explore more, high: exploit more)
      + `Q_index_map`: dictionary of stateaction to indices
      + `qval_array`: array of q-values
      + `actions_pool`: a pre-assigned numpy empty array of
           size 15 * 16

    Returns: a tuple of two things: the best action (None if no action
               can be taken), the q-value associated with that best action
               (only if choosing the best action, None otherwise)
    """
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
def top3_mixture(
    state,
    hash_state_only,
    equivalence_idx,
    valid_count,
    patient_type,
    epsilon,
    default_future_reward,
    Q_index_map,
    qval_array,
    actions_pool,
    q_value_pool
):
    """
    Chooses an action using the epsilon-greedy policy.
    Randomly chooses an action `epsilon` of the time.
    Otherwise chooses the best.

    Arguments:
      + `state` a numpy array of 45 integers {0, 1, 2} representing
           the state of the ward.
      + `hash_state_only`::::
      + `equivalence_idx`::::
      + `valid_count`::::
      + `patient_type`: the type of the patient arriving, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `epsilon`: a probability for epsilon-greedy: float between
           0 and 1 (low: explore more, high: exploit more)
      + `Q_index_map`: dictionary of stateaction to indices
      + `qval_array`: array of q-values
      + `actions_pool`: a pre-assigned numpy empty array of
           size 15 * 16

    Returns: a tuple of two things: the best action (None if no action
               can be taken), the q-value associated with that best action
               (only if choosing the best action, None otherwise)
    """
    if valid_count <= 3:
        a = choose_random_action(
            actions_pool=actions_pool,
            valid_count=valid_count
        )
        next_hash_state = hash_state_only + ward.inverse_action(a, equivalence_idx)
        return a, None, next_hash_state, equivalence_idx

    bestQ = -np.float32(np.inf)
    for i in range(valid_count):
        a = np.int64(ward.inverse_action(actions_pool[i], equivalence_idx))
        key = hash_state_only + a
        if key in Q_index_map:
            idx = Q_index_map[key]
            Q = qval_array[np.int64(idx)]
        else:
            Q = np.float32(default_future_reward)
        if Q > bestQ:
            bestQ = Q
        q_value_pool[i] = Q

    idx1, idx2, idx3 = get_top_3_indices(
        q_value_pool=q_value_pool,
        valid_count=valid_count
    )
    a1 = actions_pool[idx1]
    a2 = actions_pool[idx2]
    a3 = actions_pool[idx3]

    if np.random.random() < epsilon:
        rnd = np.random.random()
        if rnd < top3_probs[0]:
            next_hash_state = hash_state_only + ward.inverse_action(a1, equivalence_idx)
            return a1, bestQ, next_hash_state, equivalence_idx
        elif rnd < top3_probs[0] + top3_probs[1]:
            next_hash_state = hash_state_only + ward.inverse_action(a2, equivalence_idx)
            return a2, bestQ, next_hash_state, equivalence_idx
        next_hash_state = hash_state_only + ward.inverse_action(a3, equivalence_idx)
        return a3, bestQ, next_hash_state, equivalence_idx

    a = choose_random_action(
        actions_pool=actions_pool,
        valid_count=valid_count
    )
    next_hash_state = hash_state_only + ward.inverse_action(a, equivalence_idx)
    return a, bestQ, next_hash_state, equivalence_idx


@njit(cache=True)
def exploit_policy(state, patient_type, policy, actions_pool, buffer_state, fixed_mask, epsilon=1.0):
    """
    Choose an action by exploiting the policy.

    Arguments:
      + `state` a numpy array of 45 integers {0, 1, 2} representing
           the state of the ward.
      + `patient_type`: the type of the arriving patient (0, 1, or 2)
      + `policy`: the Numba typed dictionary mapping hash states to best actions
      + `actions_pool`: a pre-assigned numpy empty array of
           size 15 * 16

    Returns: the best action.
    """
    hash_state_only, equivalence_idx = ward.get_representative_hash_state(
        state=state,
        patient_type=patient_type,
        buffer_state=buffer_state
    )

    if (epsilon == 1.0):
        if hash_state_only in policy:
            a = policy[hash_state_only]
            return ward.permute_action(a, equivalence_idx)
    else:
        if np.random.random() < epsilon:
            if hash_state_only in policy:
                a = policy[hash_state_only]
                return ward.permute_action(a, equivalence_idx)

    ward.fixed_point_decision_tree(
        state=state,
        not_composed_of=ward.not_composed_of,
        fixed_mask=fixed_mask
    )

    actions_pool, valid_count = ward.get_available_actions(
        state=state,
        patient_type=patient_type,
        actions_pool=actions_pool,
        fixed_mask=fixed_mask
    )
    return choose_random_action(
        actions_pool=actions_pool,
        valid_count=valid_count
    )


@njit(cache=True)
def get_top_3_indices(q_value_pool, valid_count):
    """
    Finds the indices of the top 3 actions ranked according to their Q-values

    Arguments:
      - `q_value_pool`:
      - `valid_count`:

    Returns: the indicies of the top 3 actions, in order.
    """
    idx1, idx2, idx3 = -1, -1, -1
    val1, val2, val3 = -np.inf, -np.inf, -np.inf

    for i in range(valid_count):
        q = q_value_pool[i]
        if q > val1:
            # Shift everything down
            idx3, val3 = idx2, val2
            idx2, val2 = idx1, val1
            idx1, val1 = i, q
        elif q > val2:
            # Shift 2nd and 3rd
            idx3, val3 = idx2, val2
            idx2, val2 = i, q
        elif q > val3:
            # Update 3rd
            idx3, val3 = i, q
    return idx1, idx2, idx3