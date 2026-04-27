import rl
import ward
import pytest
import numpy as np
from numba import typed, types
import math

def test_merge_sorted_qvals():
    keys1 = np.array([1, 4, 5, 9, 11, 16])
    vals1 = np.array([0.5, 1.5, 2.0, 1.5, 4.5, 8.0])
    hits1 = np.array([1, 1, 5, 2, 3, 0])

    keys2 = np.array([2, 5, 6, 9, 10, 11, 12, 14, 16])
    vals2 = np.array([1.5, 5.0, 1.0, 1.0, 5.5, 6.0, 4.5, 1.5, 8.0])
    hits2 = np.array([3, 10, 1, 3, 2, 3, 1, 4, 0])

    keys, vals, hits = rl.merge_sorted_qvals(
        keys1=keys1,
        vals1=vals1,
        hits1=hits1,
        keys2=keys2,
        vals2=vals2,
        hits2=hits2
    )

    assert np.array_equal(keys, np.array([1, 2, 4, 5, 6, 9, 10, 11, 12, 14, 16], dtype=np.int64))
    assert np.array_equal(vals, np.array([0.5, 1.5, 1.5, 4.0, 1.0, 1.2, 5.5, 5.25, 4.5, 1.5, 8.0], dtype=np.float32))
    assert np.array_equal(hits, np.array([1, 3, 1, 15, 1, 5, 2, 6, 1, 4, 0], dtype=np.int32))


def test_get_best_future_reward():
    actions_pool = np.empty(9 + (9 * 2 * 8), dtype=np.int32)
    state = np.array(
        (3, 2, 2, 3, 2, 2, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    Qvals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    hash_state = ward.get_hash_state_only(
        state=state,
        patient_type=0,
        hash_weights=ward.hash_weights
    )
    Qvals[hash_state + 606] = -55.4
    Qvals[hash_state + 707] = -35.1
    Qvals[hash_state + 808] = -78.2

    Q = rl.get_best_future_reward(
        state=state,
        patient_type=0,
        Qvals=Qvals,
        just_chose_best=False,
        prev_best_Q=48.9,
        actions_pool=actions_pool
    )
    assert Q == -35.1

    Q = rl.get_best_future_reward(
        state=state,
        patient_type=0,
        Qvals=Qvals,
        just_chose_best=True,
        prev_best_Q=-48.9,
        actions_pool=actions_pool
    )
    assert Q == -48.9


def test_update_Q_values():
    actions_pool = np.empty(9 + (9 * 2 * 8), dtype=np.int32)
    state = np.array(
        (3, 2, 2, 3, 2, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_state = ward.get_hash_state_only(
        state=state,
        patient_type=0,
        hash_weights=ward.hash_weights
    )
    next_state = np.array(
        (3, 2, 2, 3, 2, 2, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    next_hash_state = ward.get_hash_state_only(
        state=next_state,
        patient_type=0,
        hash_weights=ward.hash_weights
    )

    Qvals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    Qvals[next_hash_state + 606] = -150.0
    Qvals[next_hash_state + 707] = -100.0
    Qvals[next_hash_state + 808] = -160.0

    hits = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    hits[hash_state + 606] = 1
    hits[hash_state + 707] = 1
    hits[hash_state + 808] = 1

    next_hash_state = rl.update_Q_values(
        hash_state=hash_state+505,
        next_state=next_state,
        next_patient_type=1,
        next_action=606,
        Qvals=Qvals,
        hits=hits,
        reward=-200,
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=False,
        prev_best_Q=-300,
        default_future_reward=-10,
        actions_pool=actions_pool
    )
    assert next_hash_state == 48504485040001606
    assert len(Qvals) == 4
    assert len(hits) == 4
    assert hits[hash_state + 505] == 1
    assert Qvals[hash_state + 505] == -145.0

    next_hash_state = rl.update_Q_values(
        hash_state=hash_state+505,
        next_state=next_state,
        next_patient_type=1,
        next_action=606,
        Qvals=Qvals,
        hits=hits,
        reward=-1000,
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=False,
        prev_best_Q=-300,
        default_future_reward=-10,
        actions_pool=actions_pool
    )
    assert next_hash_state == 48504485040001606
    assert len(Qvals) == 4
    assert len(hits) == 4
    assert hits[hash_state + 505] == 2
    assert Qvals[hash_state + 505] == -617.5

    next_hash_state = rl.update_Q_values(
        hash_state=hash_state+505,
        next_state=next_state,
        next_patient_type=1,
        next_action=606,
        Qvals=Qvals,
        hits=hits,
        reward=0,
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=True,
        prev_best_Q=-10000,
        default_future_reward=-10,
        actions_pool=actions_pool
    )
    assert next_hash_state == 48504485040001606
    assert len(Qvals) == 4
    assert len(hits) == 4
    assert hits[hash_state + 505] == 3
    assert Qvals[hash_state + 505] == -308.75 - 4500.0

def test_update_Q_values_default_future():
    actions_pool = np.empty(9 + (9 * 2 * 8), dtype=np.int32)
    state = np.array(
        (3, 2, 2, 3, 2, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_state = ward.get_hash_state_only(
        state=state,
        patient_type=0,
        hash_weights=ward.hash_weights
    )
    next_state = np.array(
        (3, 2, 2, 3, 2, 2, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    next_hash_state = ward.get_hash_state_only(
        state=next_state,
        patient_type=1,
        hash_weights=ward.hash_weights
    )

    Qvals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    hits = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    next_hash_state = rl.update_Q_values(
        hash_state=hash_state+505,
        next_state=next_state,
        next_patient_type=1,
        next_action=606,
        Qvals=Qvals,
        hits=hits,
        reward=200,
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=False,
        prev_best_Q=300,
        default_future_reward=0.2,
        actions_pool=actions_pool
    )

    assert next_hash_state == 48504485040001606
    assert len(Qvals) == 1
    assert len(hits) == 1
    assert hits[hash_state + 505] == 1
    assert Qvals[hash_state + 505] == np.float32((0.5 * 200) + (0.5 * (0.9 * (0.2 / 0.1))))


def test_initialise_qvals():
    keys1 = np.array([1, 4, 5, 9, 11, 12])
    vals1 = np.array([0.5, 1.5, 2.0, 1.5, 4.5, 6.0])
    hits1 = np.array([1, 1, 5, 2, 3, 0])
    Qvals1 = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    Hits1 = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    rl.initialise_qvals(
        keys_array=keys1,
        qval_array=vals1,
        Qvals=Qvals1,
        hits=Hits1
    )
    assert len(Qvals1) == 6
    assert len(Hits1) == 6
    assert Qvals1[1] == 0.5
    assert Qvals1[4] == 1.5
    assert Qvals1[5] == 2.0
    assert Qvals1[9] == 1.5
    assert Qvals1[11] == 4.5
    assert Qvals1[12] == 6.0
    assert Hits1[1] == 0
    assert Hits1[4] == 0
    assert Hits1[5] == 0
    assert Hits1[9] == 0
    assert Hits1[11] == 0
    assert Hits1[12] == 0

    keys2 = np.array([2, 5, 6, 9, 10, 11, 12, 14])
    vals2 = np.array([1.5, 5.0, 1.0, 1.0, 5.5, 6.0, 4.5, 1.5])
    hits2 = np.array([3, 10, 1, 3, 2, 3, 1, 4])
    Qvals2 = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    Hits2 = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    rl.initialise_qvals(
        keys_array=keys2,
        qval_array=vals2,
        Qvals=Qvals2,
        hits=Hits2
    )
    assert len(Qvals2) == 8
    assert len(Hits2) == 8
    assert Qvals2[2] == 1.5
    assert Qvals2[5] == 5.0
    assert Qvals2[6] == 1.0
    assert Qvals2[9] == 1.0
    assert Qvals2[10] == 5.5
    assert Qvals2[11] == 6.0
    assert Qvals2[12] == 4.5
    assert Qvals2[14] == 1.5
    assert Hits2[2] == 0
    assert Hits2[5] == 0
    assert Hits2[6] == 0
    assert Hits2[9] == 0
    assert Hits2[10] == 0
    assert Hits2[11] == 0
    assert Hits2[12] == 0
    assert Hits2[14] == 0


def test_initialise_policy():
    keys = np.array([22101, 22202, 22303, 33101, 33202, 33303, 44101, 44202, 44303, 55101, 66202, 66303])
    vals = np.array([  3.1,   2.1,   4.2,   7.2,   3.4,   4.3,   7.4,   8.8,   1.1,   3.2,   1.3,   1.4])
    policy = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    rl.initialise_policy(
        keys_array=keys,
        qval_array=vals,
        policy=policy
    )

    assert len(policy) == 5
    assert policy[22000] == 303
    assert policy[33000] == 101
    assert policy[44000] == 202
    assert policy[55000] == 101
    assert policy[66000] == 303

    keys = np.array([11101, 11404, 22101, 22909, 33202, 11202, 33404, 22808, 44505, 44808, 44303])
    vals = np.array([  0.1,   0.4,   0.7,   0.5,   0.3,   0.2,   0.8,   0.1,   0.9,   0.7,   0.8])
    policy = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    rl.initialise_policy(
        keys_array=keys,
        qval_array=vals,
        policy=policy
    )
    assert len(policy) == 4
    assert policy[11000] == 404
    assert policy[22000] == 101
    assert policy[33000] == 404
    assert policy[44000] == 505


def test_get_arrays_from_dicts():
    Qvals = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
    Qvals[99900096] = 50.4
    Qvals[99900097] = 100.4
    Qvals[99900098] = 60.4

    hits = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    hits[99900096] = 1
    hits[99900097] = 6
    hits[99900098] = 2

    n, k, q, h = rl.get_arrays_from_dicts(Qvals, hits)

    assert n == 3
    assert np.array_equal(k, np.array([99900096, 99900097, 99900098]))
    assert np.array_equal(q, np.array([50.4, 100.4, 60.4], dtype=np.float32))
    assert np.array_equal(h, np.array([1, 6, 2]))
