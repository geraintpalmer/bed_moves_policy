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
def choose_best_action(state, patient_type, actions_pool, valid_count, Qvals):
    """
    Chooses the best action.

    Arguments:
      + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient arriving, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `actions_pool`: a pre-assigned numpy empty array of
           size 9 + (9 * 2 * 8)
      + `valid_count`: the number of actions that are valid
      + `Qvals`: dictionary of stateaction to q-values

    Returns: an action, and the Q-value associated with that
             state-best-action pair
    """
    hash_state_only = ward.get_hash_state_only(
        state=state,
        patient_type=patient_type,
        hash_weights=ward.hash_weights
    )

    available_actions_Q = np.zeros(valid_count)
    for i in range(valid_count):
        key = hash_state_only + actions_pool[i]
        if key in Qvals:
            available_actions_Q[i] = Qvals[key]

    Qs_with_rnd = (
        available_actions_Q + (
            np.random.random(valid_count) * 10e-13
        )
    )

    idx = Qs_with_rnd.argmax()
    return actions_pool[idx], available_actions_Q[idx]


@njit(cache=True)
def choose_action(state, patient_type, epsilon, Qvals, actions_pool):
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
      + `Qvals`: a dictionary of stateaction to q-values
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
    if np.random.random() < epsilon:
        a, Qa = choose_best_action(
            state=state,
            patient_type=patient_type,
            actions_pool=actions_pool,
            valid_count=valid_count,
            Qvals=Qvals
        )
        return a, Qa

    a = choose_random_action(
        actions_pool=actions_pool,
        valid_count=valid_count
    )
    return a, None


@njit(cache=True)
def exploit_policy(state, patient_type, policy, actions_pool):
    """
    Choose an action by exploiting the policy.

    Arguments:
      + `state`: a numpy array representing the current state the ward is in
      + `patient_type`: the type of the arriving patient (0, 1, or 2)
      + `policy`: the Numba typed dictionary mapping hash states to best actions
      + `actions_pool`: a pre-assigned numpy empty array of
           size 9 + (9 * 2 * 8)

    Returns: the best action.
    """
    hash_state_only = ward.get_hash_state_only(
        state=state,
        patient_type=patient_type,
        hash_weights=ward.hash_weights
    )
    if hash_state_only in policy:
        return policy[hash_state_only]

    available_actions, valid_count = ward.get_available_actions(
        state=state,
        patient_type=patient_type,
        actions_pool=actions_pool
    )
    return choose_random_action(
        actions_pool=actions_pool,
        valid_count=valid_count
    )
