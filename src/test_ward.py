import ward
import pytest
import numpy as np

def test_get_hash_state_only():
    hash_states = [
        ward.get_hash_state_only(
            state=ward.empty_state,
            patient_type=p,
            hash_weights=ward.hash_weights
        ) for p in range(3)
    ]
    assert hash_states == [0, 10, 20]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p,
            hash_weights=ward.hash_weights
        ) for p in range(3)
    ]
    assert hash_states == [100, 110, 120]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p,
            hash_weights=ward.hash_weights
        ) for p in range(3)
    ]
    assert hash_states == [400, 410, 420]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p,
            hash_weights=ward.hash_weights
        ) for p in range(3)
    ]
    assert hash_states == [29200, 29210, 29220]

    S = np.array(
        (1, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_state = ward.get_hash_state_only(
        state=S,
        patient_type=2,
        hash_weights=ward.hash_weights
    )
    assert hash_state == 162521625229220


def test_get_hash_stateaction():
    hash_states = [
        ward.get_hash_stateaction(
            state=ward.empty_state,
            patient_type=1,
            hash_weights=ward.hash_weights, 
            action=a
        ) for a in range(9)
    ]
    assert hash_states == [10, 11, 12, 13, 14, 15, 16, 17, 18]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    hash_states = [
        ward.get_hash_stateaction(
            state=S,
            patient_type=2,
            hash_weights=ward.hash_weights,
            action=a
        ) for a in range(9)
    ]
    assert hash_states == [120, 121, 122, 123, 124, 125, 126, 127, 128]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_stateaction(
            state=S,
            patient_type=0,
            hash_weights=ward.hash_weights,
            action=a
        ) for a in range(9)
    ]
    assert hash_states == [400, 401, 402, 403, 404, 405, 406, 407, 408]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_stateaction(
            state=S,
            patient_type=1,
            hash_weights=ward.hash_weights,
            action=a
        ) for a in range(9)
    ]
    assert hash_states == [29210, 29211, 29212, 29213, 29214, 29215, 29216, 29217, 29218]


def test_get_state_action_from_hashstate():
    s, a = ward.get_state_action_from_hashstate(9996663331)
    assert s == 9996663330
    assert a == 1
    s, a = ward.get_state_action_from_hashstate(8884442225)
    assert s == 8884442220
    assert a == 5
    s, a = ward.get_state_action_from_hashstate(7773331110)
    assert s == 7773331110
    assert a == 0
    s, a = ward.get_state_action_from_hashstate(8883338883331)
    assert s == 8883338883330
    assert a == 1
    s, a = ward.get_state_action_from_hashstate(123456789123)
    assert s == 123456789120
    assert a == 3


def test_get_resource_use_per_time_unit():
    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 0

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 1

    S = np.array(
        (3, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 1

    S = np.array(
        (2, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 2

    S = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 7

    S = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         1, 0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 1, 1, 1, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 8

    S = np.array(
        (1, 0, 0, 1, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1)
    )
    assert ward.get_resource_use_per_time_unit(S) == 17


def test_get_penalty_per_time_unit():
    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 100

    S = np.array(
        (3, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (2, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 100

    S = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         1, 0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 1, 1, 1, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 200

    S = np.array(
        (1, 0, 0, 1, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 600


def test_insert_patient():
    S = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    expected_newS = np.array(
        (0, 3, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    newS = ward.insert_patient(S.copy(), 0, 1)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 1)
    )
    newS = ward.insert_patient(S.copy(), 2, 8)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 2, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    newS = ward.insert_patient(S.copy(), 1, 2)
    assert np.array_equal(newS, expected_newS)


def test_remove_patient():
    S = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    newS = ward.remove_patient(S.copy(), 0, 1)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 0, 0)
    )
    newS = ward.remove_patient(S.copy(), 2, 7)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    newS = ward.remove_patient(S.copy(), 1, 3)
    assert np.array_equal(newS, expected_newS)


def test_deteriorate_patient():
    S = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    expected_newS = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 1, 1, 1, 0)
    )
    newS = ward.deteriorate_patient(S.copy(), 1, 2)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 1, 1, 1, 0)
    )
    newS = ward.deteriorate_patient(S.copy(), 1, 3)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (0, 1, 0, 2, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    newS = ward.deteriorate_patient(S.copy(), 0, 1)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (0, 2, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 1, 2, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    newS = ward.deteriorate_patient(S.copy(), 0, 3)
    assert np.array_equal(newS, expected_newS)


def test_get_available_insert_moves():
    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_moves = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    available_moves = ward.get_available_insert_moves(S)
    assert np.array_equal(expected_moves, available_moves)

    S = np.array(
        (2, 1, 2, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 1, 0)
    )
    expected_moves = [1, 3, 4, 5, 8]
    available_moves = ward.get_available_insert_moves(S)
    assert np.array_equal(expected_moves, available_moves)
