import rl
import ward
import pytest
import numpy as np
from numba import typed, types
import math

def test_get_unique_tails():
    keys1 = np.array([1, 4, 5, 9, 11, 16], dtype=np.int64)
    vals1 = np.array([0.5, 1.5, 2.0, 1.5, 4.5, 8.0], dtype=np.float32)
    hits1 = np.array([1, 1, 5, 2, 3, 0], dtype=np.int16)

    keys2 = np.array([2, 5, 6, 9, 10, 11, 12, 14, 16], dtype=np.int64)
    vals2 = np.array([1.5, 5.0, 1.0, 1.0, 5.5, 6.0, 4.5, 1.5, 8.0], dtype=np.float32)
    hits2 = np.array([3, 10, 1, 3, 2, 3, 1, 4, 0], dtype=np.int16)

    keys, vals, hits = rl.get_unique_tails(
        keys1=keys1,
        vals1=vals1,
        hits1=hits1,
        keys2=keys2,
        vals2=vals2,
        hits2=hits2
    )

    assert np.array_equal(keys, np.array([1, 2, 4, 5, 6, 9, 10, 11, 12, 14, 16], dtype=np.int64))
    assert np.array_equal(vals, np.array([0.5, 1.5, 1.5, 4.0, 1.0, 1.2, 5.5, 5.25, 4.5, 1.5, 8.0], dtype=np.float32))
    assert np.array_equal(hits, np.array([1, 3, 1, 15, 1, 5, 2, 6, 1, 4, 0], dtype=np.int16))


def test_update_master_head_inplace():
    vals1 = np.array([0.5, 1.5, 2.0, 1.5, 4.5, 8.0], dtype=np.float32)
    hits1 = np.array([  1,   1,   5,   2,   3,   0], dtype=np.int16)
    vals2 = np.array([1.5, 5.0, 2.0, 1.0, 5.5, 8.0], dtype=np.float32)
    hits2 = np.array([  3,   9,   1,   3,   2,   0], dtype=np.int16)
    rl.update_master_head_inplace(vals1, hits1, vals2, hits2)

    assert np.array_equal(hits1, np.array([4, 10, 6, 5, 5, 0], dtype=np.int16))
    assert np.array_equal(vals1, np.array([1.25, 4.65, 2.0, 1.2, 4.9, 8.0], dtype=np.float32))


def test_update_and_merge_together():
    keys1 = np.array([111, 222, 666, 777, 999, 444, 888], dtype=np.int64)
    keys2 = np.array([111, 222, 666, 777, 999, 333, 555, 888], dtype=np.int64)
    vals1 = np.array([0.5, 1.5, 2.5, 0.5, 1.0, 1.0, 2.5], dtype=np.float32)
    vals2 = np.array([1.5, 2.0, 1.0, 0.5, 1.0, 3.0, 3.5, 1.0], dtype=np.float32)
    hits1 = np.array([  0,   1,   2,   0,   1,   4,   1], dtype=np.int16)
    hits2 = np.array([  2,   1,   1,   0,   3,   2,   1,   4], dtype=np.int16)
    M = 5

    rl.update_master_head_inplace(vals1[:M], hits1[:M], vals2[:M], hits2[:M])
    keyst, valst, hitst = rl.get_unique_tails(
        keys1=keys1[M:],
        vals1=vals1[M:],
        hits1=hits1[M:],
        keys2=keys2[M:],
        vals2=vals2[M:],
        hits2=hits2[M:]
    )
    keys = np.concatenate([keys1[:M], keyst])
    vals = np.concatenate([vals1[:M], valst])
    hits = np.concatenate([hits1[:M], hitst])

    assert np.array_equal(keys, np.array([111, 222, 666, 777, 999, 333, 444, 555, 888], dtype=np.int64))
    assert np.array_equal(hits, np.array([2, 2, 3, 0, 4, 2, 4, 1, 5], dtype=np.int16))
    assert np.array_equal(vals, np.array([1.5, 1.75, 2.0, 0.5, 1.0, 3.0, 1.0, 3.5, 1.3], dtype=np.float32))


def test_can_update_empty_arrays():
    qvals1 = np.array([], dtype=np.float32)
    hits1 = np.array([], dtype=np.int16)
    qvals2 = np.array([3.3, 4.1], dtype=np.float32)
    hits2 = np.array([5, 6], dtype=np.int16)
    M = 0
    rl.update_master_head_inplace(qvals1, hits1, qvals2, hits2)
    assert np.array_equal(qvals1, np.array([], dtype=np.float32))
    assert np.array_equal(hits1, np.array([], dtype=np.int16))

def test_get_best_future_reward():
    actions_pool = np.empty(16 * 17, dtype=np.int32)
    state = np.array(
        (1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    hash_state = ward.get_hash_state_only(
        state=state,
        patient_type=0
    )
    Q_index_map = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    Q_index_map[hash_state + np.int64(606)] = np.int32(0)
    Q_index_map[hash_state + np.int64(707)] = np.int32(1)
    Q_index_map[hash_state + np.int64(808)] = np.int32(2)
    Qvals = np.array([-55.4, -35.1, -78.2], dtype=np.float32)

    Q = rl.get_best_future_reward(
        state=state,
        patient_type=0,
        Q_index_map=Q_index_map,
        qval_array=Qvals,
        just_chose_best=False,
        prev_best_Q=np.float32(-48.9),
        actions_pool=actions_pool
    )
    assert Q == np.float32(-35.1)

    Q = rl.get_best_future_reward(
        state=state,
        patient_type=0,
        Q_index_map=Q_index_map,
        qval_array=Qvals,
        just_chose_best=True,
        prev_best_Q=np.float32(-48.9),
        actions_pool=actions_pool
    )
    assert Q == np.float32(-48.9)


def test_update_Q_values():
    actions_pool = np.empty(16 * 17, dtype=np.int32)
    state = np.array(
        (1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    hash_state = ward.get_hash_state_only(
        state=state,
        patient_type=0
    )
    next_state = np.array(
        (1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    next_hash_state = ward.get_hash_state_only(
        state=next_state,
        patient_type=0
    )

    Q_index_map = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    Q_index_map[hash_state + np.int64(606)] = np.int32(0)
    Q_index_map[hash_state + np.int64(707)] = np.int32(1)
    Q_index_map[hash_state + np.int64(808)] = np.int32(2)
    states = np.array([hash_state + 606, hash_state + 707, hash_state + 808, 0, 0, 0], dtype=np.int64)
    Qvals = np.array([-150.0, -100.0, -160.0, 0.0, 0.0, 0.0], dtype=np.float32)
    hits = np.array([1, 1, 1, 0, 0, 0], dtype=np.int16)

    next_hash_state, max_idx = rl.update_Q_values(
        hash_state=hash_state+505,
        next_state=next_state,
        next_patient_type=1,
        next_action=606,
        states_array=states,
        qval_array=Qvals,
        hits_array=hits,
        Q_index_map=Q_index_map,
        max_idx=np.int32(3),
        reward=-200,
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=False,
        prev_best_Q=np.float32(-300),
        default_future_reward=np.float32(-10),
        actions_pool=actions_pool
    )
    assert next_hash_state == 2038124953510606
    assert len(Qvals) == 6
    assert len(hits) == 6
    assert len(states) == 6
    assert len(Q_index_map) == 4
    assert max_idx == 4
    assert Q_index_map[hash_state + 505] == 3
    assert Qvals[3] == np.float32(-145.0)
    assert hits[3] == 1

    next_hash_state, max_idx = rl.update_Q_values(
        hash_state=hash_state+505,
        next_state=next_state,
        next_patient_type=1,
        next_action=606,
        states_array=states,
        qval_array=Qvals,
        hits_array=hits,
        Q_index_map=Q_index_map,
        max_idx=np.int32(4),
        reward=-1000,
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=False,
        prev_best_Q=np.float32(-300),
        default_future_reward=np.float32(-10),
        actions_pool=actions_pool
    )

    assert next_hash_state == 2038124953510606
    assert len(Qvals) == 6
    assert len(hits) == 6
    assert len(states) == 6
    assert len(Q_index_map) == 4
    assert max_idx == 4
    assert Q_index_map[hash_state + 505] == 3
    assert Qvals[3] == np.float32(-617.5)
    assert hits[3] == 2

    next_hash_state, max_idx = rl.update_Q_values(
        hash_state=hash_state+505,
        next_state=next_state,
        next_patient_type=1,
        next_action=606,
        states_array=states,
        qval_array=Qvals,
        hits_array=hits,
        Q_index_map=Q_index_map,
        max_idx=np.int32(4),
        reward=0,
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=True,
        prev_best_Q=np.float32(-10000),
        default_future_reward=np.float32(-10),
        actions_pool=actions_pool
    )

    assert next_hash_state == 2038124953510606
    assert len(Qvals) == 6
    assert len(hits) == 6
    assert len(states) == 6
    assert len(Q_index_map) == 4
    assert max_idx == 4
    assert Q_index_map[hash_state + 505] == 3
    assert Qvals[3] == np.float32(-308.75 - 4500.0)
    assert hits[3] == 3


def test_update_Q_values_default_future():
    actions_pool = np.empty(9 + (9 * 2 * 8), dtype=np.int32)
    state = np.array(
        (1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    hash_state = ward.get_hash_state_only(
        state=state,
        patient_type=0
    )
    next_state = np.array(
        (1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    next_hash_state = ward.get_hash_state_only(
        state=next_state,
        patient_type=1
    )

    Q_index_map = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    Qvals = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    hits = np.array([0, 0, 0, 0, 0], dtype=np.int16)
    states = np.array([0, 0, 0, 0, 0], dtype=np.int64)

    next_hash_state, max_idx = rl.update_Q_values(
        hash_state=hash_state+505,
        next_state=next_state,
        next_patient_type=1,
        next_action=606,
        states_array=states,
        qval_array=Qvals,
        hits_array=hits,
        Q_index_map=Q_index_map,
        max_idx=np.int32(0),
        reward=np.float32(200),
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=False,
        prev_best_Q=np.float32(-300),
        default_future_reward=np.float32(0.2),
        actions_pool=actions_pool
    )

    assert next_hash_state == 2039619173410606
    assert len(Qvals) == 5
    assert len(hits) == 5
    assert len(states) == 5
    assert len(Q_index_map) == 1
    assert max_idx == 1
    assert Q_index_map[hash_state + 505] == 0
    assert Qvals[0] == np.float32((0.5 * 200) + (0.5 * (0.9 * (0.2 / 0.1))))
    assert hits[0] == 1


def test_initialise_qvals():
    keys1 = np.array([1, 4, 5, 9, 11, 12], dtype=np.int64)
    vals1 = np.array([0.5, 1.5, 2.0, 1.5, 4.5, 6.0], dtype=np.float32)
    hits1 = np.array([1, 1, 5, 2, 3, 0], dtype=np.int16)
    Q_index_map = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    states = np.zeros(8, dtype=np.int64)
    qvals = np.zeros(8, dtype=np.float32)
    hits = np.zeros(8, dtype=np.int16)

    rl.initialise_qvals(
        initial_states_array=keys1,
        initial_qval_array=vals1,
        states_array=states,
        qval_array=qvals,
        hits_array=hits,
        Q_index_map=Q_index_map
    )

    assert np.array_equal(states, np.array([1, 4, 5, 9, 11, 12, 0, 0], dtype=np.int64))
    assert np.array_equal(qvals, np.array([0.5, 1.5, 2.0, 1.5, 4.5, 6.0, 0.0, 0.0], dtype=np.float32))
    assert np.array_equal(hits, np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16))
    assert len(Q_index_map) == 6
    for i in range(6):
        assert Q_index_map[states[i]] == i

    keys2 = np.array([2, 5, 6, 9, 10, 11, 12, 14], dtype=np.int64)
    vals2 = np.array([1.5, 5.0, 1.0, 1.0, 5.5, 6.0, 4.5, 1.5], dtype=np.float32)
    hits2 = np.array([3, 10, 1, 3, 2, 3, 1, 4], dtype=np.int16)
    Q_index_map = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    states = np.zeros(12, dtype=np.int64)
    qvals = np.zeros(12, dtype=np.float32)
    hits = np.zeros(12, dtype=np.int16)

    rl.initialise_qvals(
        initial_states_array=keys2,
        initial_qval_array=vals2,
        states_array=states,
        qval_array=qvals,
        hits_array=hits,
        Q_index_map=Q_index_map
    )
    assert np.array_equal(states, np.array([2, 5, 6, 9, 10, 11, 12, 14, 0, 0, 0, 0], dtype=np.int64))
    assert np.array_equal(qvals, np.array([1.5, 5.0, 1.0, 1.0, 5.5, 6.0, 4.5, 1.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.array_equal(hits, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16))
    assert len(Q_index_map) == 8
    for i in range(8):
        assert Q_index_map[states[i]] == i



def test_initialise_policy_dict():
    policy_keys = np.array([22000, 44000, 33000, 66000, 55000], dtype=np.int64)
    policy_actions = np.array([303, 202, 101, 303, 101], dtype=np.int16)
    policy = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    rl.initialise_policy_dict(
        keys_array=policy_keys,
        policy_array=policy_actions,
        policy=policy
    )

    assert len(policy) == 5
    assert policy[22000] == 303
    assert policy[33000] == 101
    assert policy[44000] == 202
    assert policy[55000] == 101
    assert policy[66000] == 303

    policy_keys = np.array([11000, 33000, 22000, 44000], dtype=np.int64)
    policy_actions = np.array([404, 101, 101, 505], dtype=np.int16)
    policy = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int32
    )
    rl.initialise_policy_dict(
        keys_array=policy_keys,
        policy_array=policy_actions,
        policy=policy
    )
    assert len(policy) == 4
    assert policy[11000] == 404
    assert policy[22000] == 101
    assert policy[33000] == 101
    assert policy[44000] == 505


def test_initialise_policy():
    keys = np.array([220101, 220202, 440404, 220303, 330101, 330202, 660101, 330303, 440101, 440202, 440303, 550101, 660202, 660303, 220303])
    vals = np.array([   3.1,    2.1,    2.1,    4.2,    7.2,    3.4,    0.8,    4.3,    7.4,    8.8,    1.1,    3.2,    1.3,    1.4,    1.0])
    policy_keys, policy_actions = rl.initialise_policy(
        keys_array=keys,
        qval_array=vals
    )
    assert len(policy_keys) == 5
    assert len(policy_actions) == 5
    assert np.array_equal(policy_keys, np.array([220000, 330000, 440000, 550000, 660000], dtype=np.int64))
    assert np.array_equal(policy_actions, np.array([303, 101, 202, 101, 303], dtype=np.int16))

    keys = np.array([110101, 330101, 110404, 220101, 220909, 330202, 110202, 330404, 220808, 440505, 440808, 220303, 440303])
    vals = np.array([   0.1,    1.0,    0.4,    0.7,    0.5,    0.3,    0.2,    0.8,    0.1,    0.9,    0.7,    0.5,    0.8])
    policy_keys, policy_actions = rl.initialise_policy(
        keys_array=keys,
        qval_array=vals
    )
    assert len(policy_keys) == 4
    assert len(policy_actions) == 4
    assert np.array_equal(policy_keys, np.array([110000, 220000, 330000, 440000], dtype=np.int64))
    assert np.array_equal(policy_actions, np.array([404, 101, 101, 505], dtype=np.int16))


def test_block_sort_arrays():
    states_array = np.array([ 111,  222,  333,  444,  666,  888,  999,  555,  777,   0,   0,   0,   0,   0], dtype=np.int64)
    qval_array =   np.array([-9.9, -5.5, -1.1, -4.4, -8.8, -3.3, -2.2, -7.7, -6.6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    hits_array =   np.array([   4,    1,    7,    8,    9,    2,    3,    5,    6,   0,   0,   0,   0,   0], dtype=np.int16)
    m = 5
    max_idx = 8

    max_idx2, states2, qval2, hits2 = rl.block_sort_arrays(
        states_array=states_array,
        qval_array=qval_array,
        hits_array=hits_array,
        m=m,
        max_idx=9
    )

    assert max_idx2 == 9
    assert np.array_equal(states2, np.array([111, 222, 333, 444, 666, 555, 777, 888, 999], dtype=np.int64))
    assert np.array_equal(qval2, np.array([-9.9, -5.5, -1.1, -4.4, -8.8, -7.7, -6.6, -3.3, -2.2], dtype=np.float32))
    assert np.array_equal(hits2, np.array([4, 1, 7, 8, 9, 5, 6, 2, 3], dtype=np.int16))
