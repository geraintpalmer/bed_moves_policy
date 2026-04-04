import ward
import chooser
import sim
import numpy as np
from collections import Counter
from numba import typed, types

def test_choose_random_block():
    sim.numba_seed(0)
    chosen_actions = []
    available_blocks = np.array([6, 7, 8])
    N = 100000
    for i in range(N):
        a = chooser.choose_random_block(
            available_blocks=available_blocks
        )
        chosen_actions.append(a)
    n_chosen_actions = Counter(chosen_actions)
    assert round(n_chosen_actions[6] / N, 5) == 0.33404
    assert round(n_chosen_actions[7] / N, 5) == 0.33343
    assert round(n_chosen_actions[8] / N, 5) == 0.33253


def test_choose_best_block():
    sim.numba_seed(0)
    state = np.array(
        (3, 2, 2, 3, 2, 2, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    patient_type = 1
    available_blocks = np.array([6, 7, 8])
    Qvals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    hash_state = ward.get_hash_state_only(
        state=state,
        patient_type=1,
        hash_weights=ward.hash_weights
    )

    Qvals[hash_state + 6] = 55.4
    Qvals[hash_state + 7] = 35.1
    Qvals[hash_state + 8] = 78.2
    a, Qa = chooser.choose_best_block(
        state=state,
        patient_type=1,
        available_blocks=available_blocks,
        Qvals=Qvals
    )
    assert a == 8
    assert Qa == 78.2

    Qvals[hash_state + 6] = 155.4
    Qvals[hash_state + 7] = 35.1
    Qvals[hash_state + 8] = 78.2
    a, Qa = chooser.choose_best_block(
        state=state,
        patient_type=1,
        available_blocks=available_blocks,
        Qvals=Qvals
    )
    assert a == 6
    assert Qa == 155.4

    # Test randomly chooses in a tie
    Qvals[hash_state + 6] = 0.0
    Qvals[hash_state + 7] = 0.0
    Qvals[hash_state + 8] = 0.0
    chosen_actions = []
    N = 100000
    for i in range(N):
        a, Qa = chooser.choose_best_block(
            state=state,
            patient_type=1,
            available_blocks=available_blocks,
            Qvals=Qvals
        )
        chosen_actions.append(a)
    n_chosen_actions = Counter(chosen_actions)
    assert round(n_chosen_actions[6] / N, 5) == 0.33208
    assert round(n_chosen_actions[7] / N, 5) == 0.33185
    assert round(n_chosen_actions[8] / N, 5) == 0.33607


def test_choose_arriving_block_epsilon_10():
    sim.numba_seed(0)
    S = np.array(
        (2, 1, 1, 3, 2, 2, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 1, 1)
    )
    hashS0 = ward.get_hash_stateaction(state=S, patient_type=0, action=0, hash_weights=ward.hash_weights)
    hashS1 = ward.get_hash_stateaction(state=S, patient_type=0, action=1, hash_weights=ward.hash_weights)
    hashS2 = ward.get_hash_stateaction(state=S, patient_type=0, action=2, hash_weights=ward.hash_weights)
    Qvals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    Qvals[hashS0] = 0.35
    Qvals[hashS1] = 1.56
    Qvals[hashS2] = 0.98
    a, Qa = chooser.choose_arriving_block(state=S, patient_type=0, epsilon=1.0, Qvals=Qvals)
    assert a == 1
    assert Qa == 1.56
    a, Qa = chooser.choose_arriving_block(state=S, patient_type=0, epsilon=1.0, Qvals=Qvals)
    assert a == 1
    assert Qa == 1.56
    a, Qa = chooser.choose_arriving_block(state=S, patient_type=0, epsilon=1.0, Qvals=Qvals)
    assert a == 1
    assert Qa == 1.56
    a, Qa = chooser.choose_arriving_block(state=S, patient_type=0, epsilon=1.0, Qvals=Qvals)
    assert a == 1
    assert Qa == 1.56
    a, Qa = chooser.choose_arriving_block(state=S, patient_type=0, epsilon=1.0, Qvals=Qvals)
    assert a == 1
    assert Qa == 1.56


def test_choose_arriving_block_epsilon_00():
    sim.numba_seed(0)
    S9 = np.array(
        (3, 2, 2, 3, 2, 2, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 1, 0)
    )
    S = np.array(
        (2, 1, 1, 3, 2, 2, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 1, 1)
    )
    Sfull = np.array(
        (3, 2, 2, 3, 2, 2, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 1, 1)
    )
    hashS0 = ward.get_hash_stateaction(state=S, patient_type=0, action=0, hash_weights=ward.hash_weights)
    hashS1 = ward.get_hash_stateaction(state=S, patient_type=0, action=1, hash_weights=ward.hash_weights)
    hashS2 = ward.get_hash_stateaction(state=S, patient_type=0, action=2, hash_weights=ward.hash_weights)
    Qvals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    Qvals[hashS0] = 0.35
    Qvals[hashS1] = 1.56
    Qvals[hashS2] = 0.98
    a, Qa = chooser.choose_arriving_block(state=S, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a in [0, 1, 2]
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=S, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a in [0, 1, 2]
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=S, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a in [0, 1, 2]
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=S, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a in [0, 1, 2]
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=S, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a in [0, 1, 2]
    assert Qa is None

    a, Qa = chooser.choose_arriving_block(state=S9, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a == 8
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=S9, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a == 8
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=S9, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a == 8
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=S9, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a == 8
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=S9, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a == 8
    assert Qa is None

    a, Qa = chooser.choose_arriving_block(state=Sfull, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a is None
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=Sfull, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a is None
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=Sfull, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a is None
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=Sfull, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a is None
    assert Qa is None
    a, Qa = chooser.choose_arriving_block(state=Sfull, patient_type=0, epsilon=0.0, Qvals=Qvals)
    assert a is None
    assert Qa is None


def test_choose_arriving_block_epsilon_07():
    sim.numba_seed(0)
    S = np.array(
        (2, 1, 1, 3, 2, 2, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 1, 1)
    )
    hashS0 = ward.get_hash_stateaction(state=S, patient_type=0, action=0, hash_weights=ward.hash_weights)
    hashS1 = ward.get_hash_stateaction(state=S, patient_type=0, action=1, hash_weights=ward.hash_weights)
    hashS2 = ward.get_hash_stateaction(state=S, patient_type=0, action=2, hash_weights=ward.hash_weights)
    Qvals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    Qvals[hashS0] = 0.35
    Qvals[hashS1] = 1.56
    Qvals[hashS2] = 0.98

    N = 10000
    chosen_actions = []
    for _ in range(N):
        a, Qa = chooser.choose_arriving_block(state=S, patient_type=0, epsilon=0.7, Qvals=Qvals)
        chosen_actions.append(a)
    n_chosen_actions = Counter(chosen_actions)
    assert round(n_chosen_actions[0] / N, 5) == 0.1012
    assert round(n_chosen_actions[1] / N, 5) == 0.7984
    assert round(n_chosen_actions[2] / N, 5) == 0.1004


def test_exploit_policy():
    sim.numba_seed(0)
    policy = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    policy[100] = 3
    policy[110] = 8
    policy[120] = 1

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    a = chooser.exploit_policy(state=S, patient_type=0, policy=policy)
    assert a == 3
    a = chooser.exploit_policy(state=S, patient_type=1, policy=policy)
    assert a == 8
    a = chooser.exploit_policy(state=S, patient_type=2, policy=policy)
    assert a == 1

    # Test randomly chooses if unseen
    S = np.array(
        (3, 2, 2, 3, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    chosen_actions = []
    N = 100000
    for i in range(N):
        a = chooser.exploit_policy(state=S, patient_type=0, policy=policy)
        chosen_actions.append(a)
    n_chosen_actions = Counter(chosen_actions)
    assert round(n_chosen_actions[4] / N, 5) == 0.33404
    assert round(n_chosen_actions[5] / N, 5) == 0.33343
    assert round(n_chosen_actions[6] / N, 5) == 0.33253
