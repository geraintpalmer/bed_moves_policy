import ward
import chooser
import sim
import numpy as np
from collections import Counter
from numba import typed, types

def test_choose_random_action():
    sim.numba_seed(0)
    chosen_actions = []
    actions_pool = np.array([616, 717, 826, 0, 0, 0, 0, 0, 0])
    valid_count = 3
    N = 100000
    for i in range(N):
        a = chooser.choose_random_action(
            actions_pool=actions_pool,
            valid_count=valid_count
        )
        chosen_actions.append(a)
    n_chosen_actions = Counter(chosen_actions)
    assert round(n_chosen_actions[616] / N, 5) == 0.33404
    assert round(n_chosen_actions[717] / N, 5) == 0.33343
    assert round(n_chosen_actions[826] / N, 5) == 0.33253


def test_choose_best_action():
    sim.numba_seed(0)
    state = np.array(
        (3, 2, 2, 3, 2, 2, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    patient_type = 1
    actions_pool = np.array([616, 717, 826, 0, 0, 0, 0, 0, 0])
    valid_count = 3
    Qvals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    hash_state = ward.get_hash_state_only(
        state=state,
        patient_type=1,
        hash_weights=ward.hash_weights
    )

    Qvals[hash_state + 616] = 55.4
    Qvals[hash_state + 717] = 35.1
    Qvals[hash_state + 826] = 78.2
    a, Qa = chooser.choose_best_action(
        state=state,
        patient_type=1,
        actions_pool=actions_pool,
        valid_count=valid_count,
        Qvals=Qvals
    )
    assert a == 826
    assert Qa == 78.2

    Qvals[hash_state + 616] = 155.4
    Qvals[hash_state + 717] = 35.1
    Qvals[hash_state + 826] = 78.2
    a, Qa = chooser.choose_best_action(
        state=state,
        patient_type=1,
        actions_pool=actions_pool,
        valid_count=valid_count,
        Qvals=Qvals
    )
    assert a == 616
    assert Qa == 155.4

    # Test randomly chooses in a tie
    Qvals[hash_state + 616] = 0.0
    Qvals[hash_state + 717] = 0.0
    Qvals[hash_state + 826] = 0.0
    chosen_actions = []
    N = 100000
    for i in range(N):
        a, Qa = chooser.choose_best_action(
            state=state,
            patient_type=1,
            actions_pool=actions_pool,
            valid_count=valid_count,
            Qvals=Qvals
        )
        chosen_actions.append(a)
    n_chosen_actions = Counter(chosen_actions)
    assert round(n_chosen_actions[616] / N, 5) == 0.33208
    assert round(n_chosen_actions[717] / N, 5) == 0.33185
    assert round(n_chosen_actions[826] / N, 5) == 0.33607


def test_choose_action_10():
    sim.numba_seed(0)
    actions_pool = np.empty(9 + (9 * 2 * 8), dtype=np.int32)
    S = np.array(
        (2, 1, 1, 3, 2, 2, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hashS0 = ward.get_hash_stateaction(state=S, patient_type=0, action=0, hash_weights=ward.hash_weights)
    hashS1 = ward.get_hash_stateaction(state=S, patient_type=0, action=101, hash_weights=ward.hash_weights)
    hashS2 = ward.get_hash_stateaction(state=S, patient_type=0, action=202, hash_weights=ward.hash_weights)
    Qvals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    Qvals[hashS0] = 0.35
    Qvals[hashS1] = 1.56
    Qvals[hashS2] = 0.98
    a, Qa = chooser.choose_action(state=S, patient_type=0, epsilon=1.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 101
    assert Qa == 1.56
    a, Qa = chooser.choose_action(state=S, patient_type=0, epsilon=1.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 101
    assert Qa == 1.56
    a, Qa = chooser.choose_action(state=S, patient_type=0, epsilon=1.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 101
    assert Qa == 1.56
    a, Qa = chooser.choose_action(state=S, patient_type=0, epsilon=1.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 101
    assert Qa == 1.56
    a, Qa = chooser.choose_action(state=S, patient_type=0, epsilon=1.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 101
    assert Qa == 1.56


def test_choose_action_epsilon_00():
    actions_pool = np.empty(9 + (9 * 2 * 8), dtype=np.int32)
    sim.numba_seed(0)
    S9 = np.array(
        (3, 2, 2, 3, 2, 2, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    S = np.array(
        (2, 1, 1, 3, 2, 2, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hashS0 = ward.get_hash_stateaction(state=S, patient_type=0, action=0, hash_weights=ward.hash_weights)
    hashS1 = ward.get_hash_stateaction(state=S, patient_type=0, action=101, hash_weights=ward.hash_weights)
    hashS2 = ward.get_hash_stateaction(state=S, patient_type=0, action=202, hash_weights=ward.hash_weights)
    Qvals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    Qvals[hashS0] = 0.35
    Qvals[hashS1] = 1.56
    Qvals[hashS2] = 0.98
    a, Qa = chooser.choose_action(state=S, patient_type=0, epsilon=0.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 101
    assert Qa is None
    a, Qa = chooser.choose_action(state=S, patient_type=0, epsilon=0.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 101
    assert Qa is None
    a, Qa = chooser.choose_action(state=S, patient_type=0, epsilon=0.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 202
    assert Qa is None
    a, Qa = chooser.choose_action(state=S, patient_type=0, epsilon=0.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 202
    assert Qa is None
    a, Qa = chooser.choose_action(state=S, patient_type=0, epsilon=0.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 0
    assert Qa is None

    a, Qa = chooser.choose_action(state=S9, patient_type=0, epsilon=0.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 808
    assert Qa is None
    a, Qa = chooser.choose_action(state=S9, patient_type=0, epsilon=0.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 808
    assert Qa is None
    a, Qa = chooser.choose_action(state=S9, patient_type=0, epsilon=0.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 808
    assert Qa is None
    a, Qa = chooser.choose_action(state=S9, patient_type=0, epsilon=0.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 808
    assert Qa is None
    a, Qa = chooser.choose_action(state=S9, patient_type=0, epsilon=0.0, Qvals=Qvals, actions_pool=actions_pool)
    assert a == 808
    assert Qa is None


def test_choose_action_epsilon_07():
    actions_pool = np.empty(9 + (9 * 2 * 8), dtype=np.int32)
    sim.numba_seed(0)
    S = np.array(
        (2, 1, 1, 3, 2, 2, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hashS0 = ward.get_hash_stateaction(state=S, patient_type=0, action=0, hash_weights=ward.hash_weights)
    hashS1 = ward.get_hash_stateaction(state=S, patient_type=0, action=101, hash_weights=ward.hash_weights)
    hashS2 = ward.get_hash_stateaction(state=S, patient_type=0, action=202, hash_weights=ward.hash_weights)
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
        a, Qa = chooser.choose_action(state=S, patient_type=0, epsilon=0.7, Qvals=Qvals, actions_pool=actions_pool)
        chosen_actions.append(a)
    n_chosen_actions = Counter(chosen_actions)
    assert round(n_chosen_actions[0] / N, 5) == 0.1012
    assert round(n_chosen_actions[101] / N, 5) == 0.7984
    assert round(n_chosen_actions[202] / N, 5) == 0.1004


def test_exploit_policy():
    sim.numba_seed(0)
    actions_pool = np.empty(9 + (9 * 2 * 8), dtype=np.int32)
    policy = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    policy[10000] = np.int32(303)
    policy[11000] = np.int32(808)
    policy[12000] = np.int32(101)

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    a = chooser.exploit_policy(state=S, patient_type=0, policy=policy, actions_pool=actions_pool)
    assert a == 303
    a = chooser.exploit_policy(state=S, patient_type=1, policy=policy, actions_pool=actions_pool)
    assert a == 808
    a = chooser.exploit_policy(state=S, patient_type=2, policy=policy, actions_pool=actions_pool)
    assert a == 101

    # Test randomly chooses if unseen
    S = np.array(
        (3, 2, 2, 3, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    chosen_actions = []
    N = 100000
    for i in range(N):
        a = chooser.exploit_policy(state=S, patient_type=0, policy=policy, actions_pool=actions_pool)
        chosen_actions.append(a)
    n_chosen_actions = Counter(chosen_actions)
    assert round(n_chosen_actions[404] / N, 5) == 0.33404
    assert round(n_chosen_actions[505] / N, 5) == 0.33343
    assert round(n_chosen_actions[606] / N, 5) == 0.33253
