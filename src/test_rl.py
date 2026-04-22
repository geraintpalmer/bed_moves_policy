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

    Q = rl.get_best_future_reward(
        state=state,
        patient_type=1,
        Qvals=Qvals,
        just_chose_best=False,
        prev_best_Q=48.9
    )
    assert Q == 78.2

    Q = rl.get_best_future_reward(
        state=state,
        patient_type=1,
        Qvals=Qvals,
        just_chose_best=True,
        prev_best_Q=48.9
    )
    assert Q == 48.9


def test_update_Q_values():
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
    Qvals[next_hash_state + 6] = 50.0
    Qvals[next_hash_state + 7] = 100.0
    Qvals[next_hash_state + 8] = 60.0

    hits = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    hits[hash_state + 6] = 1
    hits[hash_state + 7] = 1
    hits[hash_state + 8] = 1

    next_hash_state = rl.update_Q_values(
        hash_state=hash_state+5,
        next_state=next_state,
        next_patient_type=1,
        next_action=6,
        Qvals=Qvals,
        hits=hits,
        reward=200,
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=False,
        prev_best_Q=300,
        default_future_reward=0.1
    )

    assert next_hash_state == 485044850400016
    assert len(Qvals) == 4
    assert len(hits) == 4
    assert hits[hash_state + 5] == 1
    assert Qvals[hash_state + 5] == 145.0

    next_hash_state = rl.update_Q_values(
        hash_state=hash_state+5,
        next_state=next_state,
        next_patient_type=1,
        next_action=6,
        Qvals=Qvals,
        hits=hits,
        reward=1000,
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=False,
        prev_best_Q=300,
        default_future_reward=0.1
    )
    assert next_hash_state == 485044850400016
    assert len(Qvals) == 4
    assert len(hits) == 4
    assert hits[hash_state + 5] == 2
    assert Qvals[hash_state + 5] == 617.5

    next_hash_state = rl.update_Q_values(
        hash_state=hash_state+5,
        next_state=next_state,
        next_patient_type=1,
        next_action=6,
        Qvals=Qvals,
        hits=hits,
        reward=0,
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=True,
        prev_best_Q=10000,
        default_future_reward=0.1
    )
    assert next_hash_state == 485044850400016
    assert len(Qvals) == 4
    assert len(hits) == 4
    assert hits[hash_state + 5] == 3
    assert Qvals[hash_state + 5] == 308.75 + 4500.0

def test_update_Q_values_default_future():
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
        hash_state=hash_state+5,
        next_state=next_state,
        next_patient_type=1,
        next_action=6,
        Qvals=Qvals,
        hits=hits,
        reward=200,
        learning_rate=0.5,
        discount_factor=0.9,
        just_chose_best=False,
        prev_best_Q=300,
        default_future_reward=0.2
    )

    assert next_hash_state == 485044850400016
    assert len(Qvals) == 1
    assert len(hits) == 1
    assert hits[hash_state + 5] == 1
    assert Qvals[hash_state + 5] == np.float32((0.5 * 200) + (0.5 * (0.9 * (0.2 / 0.1))))

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
    keys = np.array([221, 222, 223, 331, 332, 333, 441, 442, 443, 551, 662, 663])
    vals = np.array([3.1, 2.1, 4.2, 7.2, 3.4, 4.3, 7.4, 8.8, 1.1, 3.2, 1.3, 1.4])
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
    assert policy[220] == 3
    assert policy[330] == 1
    assert policy[440] == 2
    assert policy[550] == 1
    assert policy[660] == 3

    keys = np.array([111, 114, 221, 229, 332, 112, 334, 228, 445, 448, 443])
    vals = np.array([0.1, 0.4, 0.7, 0.5, 0.3, 0.2, 0.8, 0.1, 0.9, 0.7, 0.8])
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
    assert policy[110] == 4
    assert policy[220] == 1
    assert policy[330] == 4
    assert policy[440] == 5


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
