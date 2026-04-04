import numpy as np
import ward
import random
from numba import njit

@njit(cache=True)
def choose_random_block(available_blocks):
    """
    Chooses a block randomly from a list of blocks.

    Arguments:
      + `available_blocks`: a list of available blocks to
           insert a patient to.

    Returns: a block to place the arriving patient.
    """
    return np.random.choice(available_blocks)


@njit(cache=True)
def choose_best_block(state, patient_type, available_blocks, Qvals):
    """
    Chooses the best action.

    Arguments:
      + `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient arriving, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `available_blocks`: a list of available blocks to
           insert a patient to.
      + `Qvals`: dictionary of stateaction to q-values

    Returns: a block to place the arriving patient, and the Q-value
               associated with the state-best-action pair
    """
    hash_state_only = ward.get_hash_state_only(
        state=state,
        patient_type=patient_type,
        hash_weights=ward.hash_weights
    )

    available_blocks_Q = np.zeros(len(available_blocks))
    for i, a in enumerate(available_blocks):
        key = hash_state_only + a
        if key in Qvals:
            available_blocks_Q[i] = Qvals[key]

    Qs_with_rnd = (
        available_blocks_Q + (
            np.random.random(available_blocks.size) * 10e-13
        )
    )

    idx = Qs_with_rnd.argmax()
    return available_blocks[idx], available_blocks_Q[idx]


@njit(cache=True)
def choose_arriving_block(state, patient_type, epsilon, Qvals):
    """
    Randomly chooses a block for an arriving patient (1-epsilon)
    of the time. Otherwise chooses the best.

    Arguments:
      + `state` a numpy array of 27 integers {0, 1, 2, 3} representing
           the state of the ward.
      + `patient_type`: the type of the patient arriving, either
           2: 'red', 1: 'amber', or 0: 'green'
      + `epsilon`: a probability, float between 0 and 1
           (low: explore more, high: exploit more)
      + `Qvals`: a dictionary of stateaction to q-values

    Returns: a tuple of two things: the best action (None if no action
               can be taken), the q-value associated with that best action
               (only if choosing the best action, None otherwise)
    """
    available_blocks = ward.get_available_insert_moves(state=state)
    if available_blocks.size == 0:
        return None, None
    if np.random.random() < epsilon:
        a, Qa = choose_best_block(
            state=state,
            patient_type=patient_type,
            available_blocks=available_blocks,
            Qvals=Qvals
        )
        return a, Qa

    a = choose_random_block(
        available_blocks=available_blocks
    )
    return a, None


@njit(cache=True)
def exploit_policy(state, patient_type, policy):
    """
    Choose an action by exploiting the policy.

    Arguments:
      + `state`: a numpy array representing the current state the ward is in
      + `patient_type`: the type of the arriving patient (0, 1, or 2)
      + `policy`: the Numba typed dictionary mapping hash states to best actions

    Returns: the best action.
    """
    hash_state_only = ward.get_hash_state_only(
        state=state,
        patient_type=patient_type,
        hash_weights=ward.hash_weights
    )
    if hash_state_only in policy:
        return policy[hash_state_only]

    available_blocks = ward.get_available_insert_moves(
        state=state
    )
    if available_blocks.size == 0:
        return None
    return choose_random_block(
        available_blocks=available_blocks
    )
