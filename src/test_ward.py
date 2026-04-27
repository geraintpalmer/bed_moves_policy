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
    assert hash_states == [000, 1000, 2000]

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
    assert hash_states == [10000, 11000, 12000]

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
    assert hash_states == [40000, 41000, 42000]

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
    assert hash_states == [2920000, 2921000, 2922000]

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
    assert hash_state == 16252162522922000


def test_get_hash_stateaction():
    hash_states = [
        ward.get_hash_stateaction(
            state=ward.empty_state,
            patient_type=1,
            hash_weights=ward.hash_weights, 
            action=((100 * a) +  (10 * 1) + a)
        ) for a in range(9)
    ]
    assert hash_states == [1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818]

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
            action=np.array(a)
        ) for a in [20, 121, 222, 323, 424, 525, 626, 727]
    ]
    assert hash_states == [12020, 12121, 12222, 12323, 12424, 12525, 12626, 12727]

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
            action=np.array(a)
        ) for a in [ 20, 121, 222, 323, 424, 525, 626, 727,
                    800, 801, 802, 803, 804, 805, 806, 807]
    ]
    assert hash_states == [
        40020, 40121, 40222, 40323, 40424, 40525, 40626, 40727,
        40800, 40801, 40802, 40803, 40804, 40805, 40806, 40807
    ]

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
            action=np.array(a)
        ) for a in [ 10, 111, 212, 313, 414, 515,
                    600, 601, 602, 603, 604, 605,
                    700, 701, 702, 703, 704, 705,
                    800, 801, 802, 803, 804, 805]
    ]
    assert hash_states == [
        2921010, 2921111, 2921212, 2921313, 2921414, 2921515,
        2921600, 2921601, 2921602, 2921603, 2921604, 2921605,
        2921700, 2921701, 2921702, 2921703, 2921704, 2921705,
        2921800, 2921801, 2921802, 2921803, 2921804, 2921805
    ]


def test_get_state_action_from_hashstate():
    s, a = ward.get_state_action_from_hashstate(999666333122)
    assert s == 999666333000
    assert a == 122
    s, a = ward.get_state_action_from_hashstate(888444222505)
    assert s == 888444222000
    assert a == 505
    s, a = ward.get_state_action_from_hashstate(777333111081)
    assert s == 777333111000
    assert a == 81
    s, a = ward.get_state_action_from_hashstate(888333888333114)
    assert s == 888333888333000
    assert a == 114
    s, a = ward.get_state_action_from_hashstate(12345678912356)
    assert s == 12345678912000
    assert a == 356


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


def test_get_move_penalty():
    move_penalties = np.array(
        [
            [5.0, 6.0, 7.0],
            [5.5, 6.5, 7.5]
        ]
    )
    assert ward.get_move_penalty(0, 1, 0, 0, move_penalties, ward.adjacency_matrix) == 5.0
    assert ward.get_move_penalty(0, 7, 0, 0, move_penalties, ward.adjacency_matrix) == 5.5
    assert ward.get_move_penalty(0, 1, 1, 0, move_penalties, ward.adjacency_matrix) == 6.0
    assert ward.get_move_penalty(0, 7, 1, 0, move_penalties, ward.adjacency_matrix) == 6.5
    assert ward.get_move_penalty(0, 1, 2, 0, move_penalties, ward.adjacency_matrix) == 7.0
    assert ward.get_move_penalty(0, 7, 2, 0, move_penalties, ward.adjacency_matrix) == 7.5
    assert ward.get_move_penalty(5, 5, 1, 1, move_penalties, ward.adjacency_matrix) == 0.0


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


def test_move_patient():
    S = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 2, 0, 0, 0, 0, 1,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    newS = ward.move_patient(S.copy(), 0, 8, 1)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 1, 1, 0, 0)
    )
    newS = ward.move_patient(S.copy(), 2, 3, 7)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    newS = ward.move_patient(S.copy(), 1, 4, 3)
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


def test_get_available_actions():
    S = np.array(
        (3, 2, 2, 3, 1, 2, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    actions_pool = np.empty(9 + (9 * 2 * 8), dtype=np.int32)
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([414, 4, 104, 204, 304, 504, 604, 704, 804], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([424, 4, 104, 204, 304, 504, 604, 704, 804], dtype=np.int32))

    S = np.array(
        (3, 0, 0, 1, 2, 2, 0, 1, 1,
         0, 2, 1, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([606, 116, 216, 226, 316], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([616, 6, 226, 306, 406, 506, 706, 806], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([626, 6, 116, 216, 306, 316, 406, 506, 706, 806], dtype=np.int32))

    S = np.array(
        (3, 2, 2, 3, 2, 2, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([], dtype=np.int32))


def test_find_idx_of_patient_to_move():
    patients_blocks = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 8, 8])
    patients_types =  np.array([0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 1, 1, 0, 1, 2])
    assert 8 == ward.find_idx_of_patient_to_move(block=3, patient_type=2, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 0 == ward.find_idx_of_patient_to_move(block=0, patient_type=0, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 16 == ward.find_idx_of_patient_to_move(block=8, patient_type=2, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 5 == ward.find_idx_of_patient_to_move(block=2, patient_type=0, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 6 == ward.find_idx_of_patient_to_move(block=2, patient_type=1, patients_blocks=patients_blocks, patients_types=patients_types)
