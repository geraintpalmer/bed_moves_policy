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
    buffer_state = np.zeros(48, dtype=np.int64)
    state = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    patient_type = 1
    actions_pool = np.array([1212, 1313, 1515, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    valid_count = 3
    Q_index_map = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    hash_state_only, equivalence_idx = ward.get_representative_hash_state(
        state=state,
        patient_type=patient_type,
        buffer_state=buffer_state
    )

    Q_index_map[hash_state_only + np.int64(1212)] = np.int32(0)
    Q_index_map[hash_state_only + np.int64(1313)] = np.int32(1)
    Q_index_map[hash_state_only + np.int64(1515)] = np.int32(2)
    Qvals = np.array([55.4, 35.1, 78.2], dtype=np.float32)
    a, Qa = chooser.choose_best_action(
        state=state,
        hash_state_only=hash_state_only,
        equivalence_idx=equivalence_idx,
        patient_type=patient_type,
        actions_pool=actions_pool,
        valid_count=valid_count,
        Q_index_map=Q_index_map,
        qval_array=Qvals
    )
    assert a == 1515
    assert Qa == np.float32(78.2)

    Q_index_map[hash_state_only + np.int64(1212)] = np.int32(0)
    Q_index_map[hash_state_only + np.int64(1313)] = np.int32(1)
    Q_index_map[hash_state_only + np.int64(1515)] = np.int32(2)
    Qvals = np.array([155.4, 35.1, 78.2], dtype=np.float32)

    a, Qa = chooser.choose_best_action(
        state=state,
        hash_state_only=hash_state_only,
        equivalence_idx=equivalence_idx,
        patient_type=patient_type,
        actions_pool=actions_pool,
        valid_count=valid_count,
        Q_index_map=Q_index_map,
        qval_array=Qvals
    )
    assert a == 1212
    assert Qa == np.float32(155.4)

    # Test randomly chooses in a tie
    Q_index_map[hash_state_only + np.int64(1212)] = np.int32(0)
    Q_index_map[hash_state_only + np.int64(1313)] = np.int32(1)
    Q_index_map[hash_state_only + np.int64(1515)] = np.int32(2)
    Qvals = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    chosen_actions = []
    N = 100000
    for i in range(N):
        a, Qa = chooser.choose_best_action(
            state=state,
            hash_state_only=hash_state_only,
            equivalence_idx=equivalence_idx,
            patient_type=patient_type,
            actions_pool=actions_pool,
            valid_count=valid_count,
            Q_index_map=Q_index_map,
            qval_array=Qvals
        )
        chosen_actions.append(a)
    n_chosen_actions = Counter(chosen_actions)
    assert round(n_chosen_actions[1212] / N, 5) == 0.33208
    assert round(n_chosen_actions[1313] / N, 5) == 0.33185
    assert round(n_chosen_actions[1515] / N, 5) == 0.33607


def test_choose_action_10():
    sim.numba_seed(0)
    actions_pool = np.empty(16 * 17, dtype=np.int32)
    buffer_state = np.zeros(48, dtype=np.int64)
    S = np.array(
        (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    hashS0, equivalence_idx0 = ward.get_hash_stateaction(state=S, patient_type=0, action=0, buffer_state=buffer_state)
    hashS1, equivalence_idx1 = ward.get_hash_stateaction(state=S, patient_type=0, action=101, buffer_state=buffer_state)
    hashS2, equivalence_idx2 = ward.get_hash_stateaction(state=S, patient_type=0, action=202, buffer_state=buffer_state)
    Q_index_map = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    Q_index_map[hashS0] = np.int32(0)
    Q_index_map[hashS1] = np.int32(1)
    Q_index_map[hashS2] = np.int32(2)
    Qvals = np.array([0.35, 1.56, 0.98], dtype=np.float32)
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S, patient_type=0, epsilon=1.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 101
    assert Qa == np.float32(1.56)
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S, patient_type=0, epsilon=1.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 101
    assert Qa == np.float32(1.56)
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S, patient_type=0, epsilon=1.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 101
    assert Qa == np.float32(1.56)
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S, patient_type=0, epsilon=1.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 101
    assert Qa == np.float32(1.56)
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S, patient_type=0, epsilon=1.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 101
    assert Qa == np.float32(1.56)


def test_choose_action_epsilon_00():
    actions_pool = np.empty(16 * 17, dtype=np.int32)
    buffer_state = np.zeros(48, dtype=np.int64)
    sim.numba_seed(0)
    S9 = np.array(
        (1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    S = np.array(
        (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    hashS0, equivalence_idx0 = ward.get_hash_stateaction(state=S, patient_type=0, action=0, buffer_state=buffer_state)
    hashS1, equivalence_idx1 = ward.get_hash_stateaction(state=S, patient_type=0, action=101, buffer_state=buffer_state)
    hashS2, equivalence_idx2 = ward.get_hash_stateaction(state=S, patient_type=0, action=202, buffer_state=buffer_state)
    Q_index_map = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    Q_index_map[hashS0] = np.int32(0)
    Q_index_map[hashS1] = np.int32(1)
    Q_index_map[hashS2] = np.int32(2)
    Qvals = np.array([0.35, 1.56, 0.98], dtype=np.float32)
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([0, 101, 202], dtype=np.int32))
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S, patient_type=0, epsilon=0.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 101
    assert Qa is None
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S, patient_type=0, epsilon=0.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 101
    assert Qa is None
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S, patient_type=0, epsilon=0.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 202
    assert Qa is None
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S, patient_type=0, epsilon=0.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 202
    assert Qa is None
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S, patient_type=0, epsilon=0.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 0
    assert Qa is None

    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S9, patient_type=0, epsilon=0.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 808
    assert Qa is None
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S9, patient_type=0, epsilon=0.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 808
    assert Qa is None
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S9, patient_type=0, epsilon=0.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 808
    assert Qa is None
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S9, patient_type=0, epsilon=0.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 808
    assert Qa is None
    a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S9, patient_type=0, epsilon=0.0, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 808
    assert Qa is None


def test_choose_action_epsilon_07():
    actions_pool = np.empty(16 * 17, dtype=np.int32)
    buffer_state = np.zeros(48, dtype=np.int64)
    sim.numba_seed(0)
    S = np.array(
        (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    hashS0, equivalence_idx0 = ward.get_hash_stateaction(state=S, patient_type=0, action=0, buffer_state=buffer_state)
    hashS1, equivalence_idx1 = ward.get_hash_stateaction(state=S, patient_type=0, action=101, buffer_state=buffer_state)
    hashS2, equivalence_idx2 = ward.get_hash_stateaction(state=S, patient_type=0, action=202, buffer_state=buffer_state)
    Q_index_map = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    Q_index_map[hashS0] = np.int32(0)
    Q_index_map[hashS1] = np.int32(1)
    Q_index_map[hashS2] = np.int32(2)
    Qvals = np.array([0.35, 1.56, 0.98], dtype=np.float32)

    N = 10000
    chosen_actions = []
    for _ in range(N):
        a, Qa, next_hash_state, next_equiv_idx = chooser.choose_action(state=S, patient_type=0, epsilon=0.7, Q_index_map=Q_index_map, qval_array=Qvals, actions_pool=actions_pool, buffer_state=buffer_state)
        chosen_actions.append(a)
    n_chosen_actions = Counter(chosen_actions)
    assert round(n_chosen_actions[0] / N, 5) == 0.1012
    assert round(n_chosen_actions[101] / N, 5) == 0.7984
    assert round(n_chosen_actions[202] / N, 5) == 0.1004


def test_exploit_policy():
    sim.numba_seed(0)
    buffer_state = np.zeros(48, dtype=np.int64)
    actions_pool = np.empty(16 * 17, dtype=np.int32)
    policy = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    policy[100000] = np.int32(303)
    policy[110000] = np.int32(808)
    policy[120000] = np.int32(101)

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    a = chooser.exploit_policy(state=S, patient_type=0, policy=policy, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 303
    a = chooser.exploit_policy(state=S, patient_type=1, policy=policy, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 808
    a = chooser.exploit_policy(state=S, patient_type=2, policy=policy, actions_pool=actions_pool, buffer_state=buffer_state)
    assert a == 101

    # Test randomly chooses if unseen
    S = np.array(
        (1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    chosen_actions = []
    N = 100000
    for i in range(N):
        a = chooser.exploit_policy(state=S, patient_type=0, policy=policy, actions_pool=actions_pool, buffer_state=buffer_state)
        chosen_actions.append(a)
    n_chosen_actions = Counter(chosen_actions)
    assert round(n_chosen_actions[404] / N, 5) == 0.33404
    assert round(n_chosen_actions[505] / N, 5) == 0.33343
    assert round(n_chosen_actions[606] / N, 5) == 0.33253
