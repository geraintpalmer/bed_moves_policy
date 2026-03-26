import bedmoves
import pytest
import numpy as np
import ciw
import math
import pandas as pd

class FakeActionChooser:
    just_chose_best = False
class FakeSimulation:
    action_chooser = FakeActionChooser()
    rng = np.random.default_rng(seed=0)



def test_get_resource_use_per_time_unit():
    S = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    assert bedmoves.get_resource_use_per_time_unit(S) == 0

    S = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0, 0)
        )
    )
    assert bedmoves.get_resource_use_per_time_unit(S) == 1

    S = np.array(
        (
            (3, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    assert bedmoves.get_resource_use_per_time_unit(S) == 1

    S = np.array(
        (
            (2, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    assert bedmoves.get_resource_use_per_time_unit(S) == 2

    S = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    assert bedmoves.get_resource_use_per_time_unit(S) == 7

    S = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 1, 1, 0)
        )
    )
    assert bedmoves.get_resource_use_per_time_unit(S) == 8

    S = np.array(
        (
            (1, 0, 0, 1, 0, 0, 0, 0, 0),
            (1, 1, 1, 1, 1, 1, 0, 0, 0),
            (1, 1, 1, 1, 1, 1, 1, 1, 1)
        )
    )
    assert bedmoves.get_resource_use_per_time_unit(S) == 17



def test_get_penalty_per_time_unit():
    S = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    assert bedmoves.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0, 0)
        )
    )
    assert bedmoves.get_penalty_per_time_unit(S, 100) == 100

    S = np.array(
        (
            (3, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    assert bedmoves.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (
            (2, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    assert bedmoves.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    assert bedmoves.get_penalty_per_time_unit(S, 100) == 100

    S = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 1, 1, 0)
        )
    )
    assert bedmoves.get_penalty_per_time_unit(S, 100) == 200

    S = np.array(
        (
            (1, 0, 0, 1, 0, 0, 0, 0, 0),
            (1, 1, 1, 1, 1, 1, 0, 0, 0),
            (1, 1, 1, 1, 1, 1, 1, 1, 1)
        )
    )
    assert bedmoves.get_penalty_per_time_unit(S, 100) == 600


def test_get_move_penalty():
    assert bedmoves.get_move_penalty(1, 2, 10, 20) == 10
    assert bedmoves.get_move_penalty(1, 1, 10, 20) == 0
    assert bedmoves.get_move_penalty(1, 3, 10, 20) == 20

def test_move_patient():
    S = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    expected_newS = np.array(
        (
            (0, 1, 1, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    newS = bedmoves.move_patient(S.copy(), 0, 1, 2)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 1, 1, 0, 0)
        )
    )
    newS = bedmoves.move_patient(S.copy(), 2, 7, 0)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 1, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    newS = bedmoves.move_patient(S.copy(), 1, 3, 1)
    assert np.array_equal(newS, expected_newS)


def test_insert_patient():
    S = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    expected_newS = np.array(
        (
            (0, 3, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    newS = bedmoves.insert_patient(S.copy(), 0, 1)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 1)
        )
    )
    newS = bedmoves.insert_patient(S.copy(), 2, 8)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 2, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    newS = bedmoves.insert_patient(S.copy(), 1, 2)
    assert np.array_equal(newS, expected_newS)


def test_remove_patient():
    S = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    expected_newS = np.array(
        (
            (0, 1, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    newS = bedmoves.remove_patient(S.copy(), 0, 1)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 0, 0)
        )
    )
    newS = bedmoves.remove_patient(S.copy(), 2, 7)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    newS = bedmoves.remove_patient(S.copy(), 1, 3)
    assert np.array_equal(newS, expected_newS)


def test_get_available_insert_moves():
    S = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    expected_moves = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    available_moves = bedmoves.get_available_insert_moves(S)
    assert np.array_equal(expected_moves, available_moves)

    S = np.array(
        (
            (2, 1, 2, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 0)
        )
    )
    expected_moves = [1, 3, 4, 5, 8]
    available_moves = bedmoves.get_available_insert_moves(S)
    assert np.array_equal(expected_moves, available_moves)



def test_get_available_moves():
    S = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 1, 1, 0)
        )
    )
    expected_moves = [
        (0, 1, 0), (0, 1, 2), (0, 1, 4), (0, 1, 5), (0, 1, 8),
        (0, 3, 0), (0, 3, 2), (0, 3, 4), (0, 3, 5), (0, 3, 8),
        (1, 0, 2), (1, 0, 4), (1, 0, 5), (1, 0, 8),
        (1, 2, 0), (1, 2, 4), (1, 2, 5), (1, 2, 8),
        (2, 3, 0), (2, 3, 2), (2, 3, 4), (2, 3, 5), (2, 3, 8),
        (2, 5, 0), (2, 5, 2), (2, 5, 4), (2, 5, 8),
        (2, 6, 0), (2, 6, 2), (2, 6, 4), (2, 6, 5), (2, 6, 8),
        (2, 7, 0), (2, 7, 2), (2, 7, 4), (2, 7, 5), (2, 7, 8)
    ]

    available_moves = bedmoves.get_available_moves(S)
    assert expected_moves == available_moves


    S = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    available_moves = bedmoves.get_available_moves(S)
    assert available_moves == []


def test_combine_arrays():
    keys1 = [1111, 1112, 2221]
    qval1 = [150.0, 200.0, 300.0]
    hits1 = [1, 2, 3]

    keys2 = [1111, 1113, 2221, 3332]
    qval2 = [50.0, 250.0, 500.0, 100.0]
    hits2 = [1, 2, 1, 4]


    keys, qval, hits = bedmoves.combine_arrays(
        [keys1, keys2], [qval1, qval2], [hits1, hits2]
    )
    expected_keys = [1111, 1112, 1113, 2221, 3332]
    expected_qval = [100.0, 200.0, 250.0, 350.0, 100.0]
    expected_hits = [2, 2, 2, 4, 4]

    assert np.array_equal(keys, expected_keys)
    assert np.array_equal(qval, expected_qval)
    assert np.array_equal(hits, expected_hits)


def test_epsilonhard_00():
    RC = bedmoves.EpsilonHard(epsilon=0.0, QLearning=None)
    S9 = np.array(
        (
            (3, 2, 2, 3, 2, 2, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 0)
        )
    )

    S123 = np.array(
        (
            (2, 1, 1, 3, 2, 2, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 1)
        )
    )

    Sfull = np.array(
        (
            (3, 2, 2, 3, 2, 2, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 1)
        )
    )

    ciw.seed(0)
    assert RC.choose_arriving_block(S9, 0) == 8
    assert RC.choose_arriving_block(S9, 0) == 8
    assert RC.choose_arriving_block(S9, 0) == 8
    assert RC.choose_arriving_block(S9, 0) == 8
    assert RC.choose_arriving_block(S9, 0) == 8
    assert RC.choose_arriving_block(S9, 0) == 8

    ciw.seed(0)
    assert RC.choose_arriving_block(S123, 0) in [0, 1, 2]
    assert RC.choose_arriving_block(S123, 0) in [0, 1, 2]
    assert RC.choose_arriving_block(S123, 0) in [0, 1, 2]
    assert RC.choose_arriving_block(S123, 0) in [0, 1, 2]
    assert RC.choose_arriving_block(S123, 0) in [0, 1, 2]
    assert RC.choose_arriving_block(S123, 0) in [0, 1, 2]

    ciw.seed(0)
    assert RC.choose_arriving_block(Sfull, 0) == None
    assert RC.choose_arriving_block(Sfull, 0) == None
    assert RC.choose_arriving_block(Sfull, 0) == None
    assert RC.choose_arriving_block(Sfull, 0) == None
    assert RC.choose_arriving_block(Sfull, 0) == None
    assert RC.choose_arriving_block(Sfull, 0) == None


def test_epsilonhard_10():
    S123 = np.array(
        (
            (2, 1, 1, 3, 2, 2, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 1)
        )
    )
    hashS123_1 = bedmoves.get_hash_stateaction(state=S123, patient_type=0, action=0)
    hashS123_2 = bedmoves.get_hash_stateaction(state=S123, patient_type=0, action=1)
    hashS123_3 = bedmoves.get_hash_stateaction(state=S123, patient_type=0, action=2)
    
    initial_qvals = np.array(
        [
            (hashS123_1, 0.35, 34),
            (hashS123_2, 1.56, 12),
            (hashS123_3, 0.98, 55)
        ], dtype=[('Key', '<i8'), ('Q', '<f4'), ('Hits', '<i4')]
    )

    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=2.0,
        initial_Qvalues=initial_qvals
    )
    EH = bedmoves.EpsilonHard(epsilon=1.0, QLearning=Q)
    EH.rng = np.random.default_rng(seed=0)
    ciw.seed(0)
    assert EH.choose_arriving_block(S123, 0) == 1
    assert EH.choose_arriving_block(S123, 0) == 1
    assert EH.choose_arriving_block(S123, 0) == 1
    assert EH.choose_arriving_block(S123, 0) == 1
    assert EH.choose_arriving_block(S123, 0) == 1
    assert EH.choose_arriving_block(S123, 0) == 1


def test_epsilonhard_07():
    ## Choosing epsilon as 07 should result in action 2 being chosen
    ## 4/5 of the time (7/10 of the time as the best, and (3/10 * 1/3)
    ## of the time randomly)
    S123 = np.array(
        (
            (2, 1, 1, 3, 2, 2, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 1)
        )
    )
    hashS123_1 = bedmoves.get_hash_stateaction(state=S123, patient_type=0, action=0)
    hashS123_2 = bedmoves.get_hash_stateaction(state=S123, patient_type=0, action=1)
    hashS123_3 = bedmoves.get_hash_stateaction(state=S123, patient_type=0, action=2)

    initial_qvals = np.array(
        [
            (hashS123_1, 0.35, 34),
            (hashS123_2, 1.56, 12),
            (hashS123_3, 0.98, 55)
        ], dtype=[('Key', '<i8'), ('Q', '<f4'), ('Hits', '<i4')]
    )

    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=2.0,
        initial_Qvalues=initial_qvals
    )
    EH = bedmoves.EpsilonHard(epsilon=0.7, QLearning=Q)
    EH.rng = np.random.default_rng(seed=0)
    ciw.seed(0)
    choices = [EH.choose_arriving_block(S123, 0) for _ in range(1000)]
    assert sum(c == 1 for c in choices) == 807



def test_QLearning_init():
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=1.0
        )

    assert Q.learning_rate == 0.5
    assert Q.discount_factor == 0.9
    assert Q.transform_parameter == 1.0
    assert Q.previous_cost == 0.0
    assert Q.Qvals == {}
    assert Q.hash_state == None

    FS = FakeSimulation()
    Q.attach_simulation(FS)
    assert Q.simulation == FS

def test_QLearning_initial_Qvalues():
    S1 = 911100000000000000
    S2 = 911100000000000001
    S3 = 900000000000000000
    states = [int(str(S1) + '4'), int(str(S1) + '5'), int(str(S2) + '4'), int(str(S2) + '5'), int(str(S2) + '6'), int(str(S3) + '2')]

    initial_qvals = np.array(
        [
            (int(str(S1) + '4'), 1.2, 34),
            (int(str(S1) + '5'), 1.8, 12),
            (int(str(S2) + '4'), 1.1, 55),
            (int(str(S2) + '5'), 0.7, 2),
            (int(str(S2) + '6'), 7.1, 5),
            (int(str(S3) + '2'), 3.1, 6),
        ], dtype=[('Key', '<i8'), ('Q', '<f8'), ('Hits', '<i4')]
    )

    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=1.0,
        initial_Qvalues=initial_qvals
        )

    assert Q.learning_rate == 0.5
    assert Q.discount_factor == 0.9
    assert Q.transform_parameter == 1.0
    assert Q.previous_cost == 0.0
    assert Q.hash_state == None

    assert len(Q.Qvals.keys()) == 6
    assert set([k for k in Q.Qvals.keys()]) == set(states)
    assert np.array_equal([Q.getQ(k) for k in states], [1.2, 1.8, 1.1, 0.7, 7.1, 3.1])
    assert np.array_equal([Q.getQtuple(k).hits for k in states], [0, 0, 0, 0, 0, 0])


def test_QLearning_hashstate():
    S = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 1, 1, 0)
        )
    )
    assert bedmoves.get_hash_state_only(state=S, patient_type=1) == 44893300107210
    assert bedmoves.get_hash_stateaction(state=S, patient_type=1, action=2) == 44893300107212


def test_transform_cost():
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=1.0
        )
    cost1 = -math.log(7)
    cost2 = -math.log(2)
    cost3 = -math.log(15)

    assert np.isclose(Q.transform_cost(cost1), 7)
    assert np.isclose(Q.transform_cost(cost2), 2)
    assert np.isclose(Q.transform_cost(cost3), 15)


def test_update_Q_values():
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=1.0
        )
    S = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    Q.hash_state = bedmoves.get_hash_stateaction(state=S, patient_type=1, action=1)
    Q.previous_cost = 90

    FS = FakeSimulation()
    FS.overall_cost = 100
    FS.now = 40
    Q.attach_simulation(FS)

    next_state = np.array(
        (
            (1, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    next_patient_type = 2
    next_action = 5

    Q.update_Q_values(next_state, next_patient_type, next_action)

    assert Q.hash_state == bedmoves.get_hash_stateaction(state=next_state, patient_type=2, action=5)
    R = math.exp(-Q.transform_parameter * 10)
    assert Q.Qvals[bedmoves.get_hash_stateaction(state=S, patient_type=1, action=1)].Q == 0.5 * R
    assert Q.Qvals[bedmoves.get_hash_stateaction(state=S, patient_type=1, action=1)].hits == 1


def test_BedMovesSimulation_init():
    S = bedmoves.BedMoveSimulation(
        arrival_distributions=[
            ciw.dists.Deterministic(5),
            ciw.dists.Deterministic(9),
            ciw.dists.Deterministic(11)
        ],
        los_distributions=[
            ciw.dists.Deterministic(1),
            ciw.dists.Deterministic(3),
            ciw.dists.Deterministic(7)
        ],
        action_chooser=bedmoves.EpsilonHard(epsilon=0.0, QLearning=None),
        isolation_penalty=2.0,
        adjacent_move_penalty=2.0,
        nonadjacent_move_penalty=2.0,
        QLearning=bedmoves.QLearning(0.5, 0.9, 1.0),
        seed=0
    )

    assert len(S.arrival_distributions) == 3
    assert len(S.los_distributions) == 3
    assert str(S.action_chooser) == "EpsilonHard-0.0"
    assert S.next_arrivals == {0: 5, 1: 9, 2: 11}
    assert str(S.QLearning) == "QLearning"
    assert S.prev_now == 0.0
    assert S.now == 0.0
    assert S.overall_cost == 0.0
    assert S.isolation_penalty == 2.0
    assert S.adjacent_move_penalty == 2.0
    assert S.nonadjacent_move_penalty == 2.0
    assert len(S.patients.patient_types) == 17
    assert len(S.patients.los) == 17
    assert len(S.patients.exit_dates) == 17
    assert len(S.patients.blocks) == 17
    assert np.min(S.patients.patient_types) == -1
    assert np.max(S.patients.patient_types) == -1
    assert np.min(S.patients.blocks) == -1
    assert np.max(S.patients.blocks) == -1
    assert np.min(S.patients.los) == np.inf
    assert np.max(S.patients.los) == np.inf
    assert np.min(S.patients.exit_dates) == np.inf
    assert np.max(S.patients.exit_dates) == np.inf


def test_BedMovesSimulation_nextarrival():
    S = bedmoves.BedMoveSimulation(
        arrival_distributions=[
            ciw.dists.Deterministic(5),
            ciw.dists.Deterministic(9),
            ciw.dists.Deterministic(11)
        ],
        los_distributions=[
            ciw.dists.Deterministic(1),
            ciw.dists.Deterministic(3),
            ciw.dists.Deterministic(7)
        ],
        action_chooser=bedmoves.EpsilonHard(epsilon=0.0, QLearning=None),
        isolation_penalty=2.0,
        adjacent_move_penalty=2.0,
        nonadjacent_move_penalty=2.0,
        QLearning=bedmoves.QLearning(0.5, 0.9, 1.0),
        seed=0
    )

    assert S.next_arrivals == {0: 5, 1: 9, 2: 11}
    next_date, next_type =  S.find_next_arrival_date()
    assert next_date == 5
    assert next_type == 0

    S.next_arrivals = {0: 41.2, 1: 9.4, 2: 34.1}
    next_date, next_type =  S.find_next_arrival_date()
    assert next_date == 9.4
    assert next_type == 1

    S.next_arrivals = {0: 0.23, 1: 0.87, 2: 0.18}
    next_date, next_type =  S.find_next_arrival_date()
    assert next_date == 0.18
    assert next_type == 2


def test_BedMovesSimulation_nextexit():
    S = bedmoves.BedMoveSimulation(
        arrival_distributions=[
            ciw.dists.Deterministic(5),
            ciw.dists.Deterministic(9),
            ciw.dists.Deterministic(11)
        ],
        los_distributions=[
            ciw.dists.Deterministic(1),
            ciw.dists.Deterministic(3),
            ciw.dists.Deterministic(7)
        ],
        action_chooser=bedmoves.EpsilonHard(epsilon=0.0, QLearning=None),
        isolation_penalty=2.0,
        adjacent_move_penalty=2.0,
        nonadjacent_move_penalty=2.0,
        QLearning=bedmoves.QLearning(0.5, 0.9, 1.0),
        seed=0
    )

    next_exit, next_patient_idx = S.find_next_exit_date()
    assert next_exit == float('inf')
    assert next_patient_idx == None

    S.patients = bedmoves.Patients(
        patient_types=np.array([0, 0, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
        los=np.array([31.5, 5.6, 22.2, 9.6, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
        exit_dates=np.array([33.9, 11.7, 23.3, 16.9, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
        blocks=np.array([1, 1, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
        free_indices=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    )

    next_exit, next_patient_idx = S.find_next_exit_date()
    assert next_exit == 11.7
    assert next_patient_idx == 1


def test_BedMovesSimulation_inflict_cost():
    # Resource use 7 per time unit, penalty 1:
    S = bedmoves.BedMoveSimulation(
        arrival_distributions=[
            ciw.dists.Deterministic(5),
            ciw.dists.Deterministic(9),
            ciw.dists.Deterministic(11)
        ],
        los_distributions=[
            ciw.dists.Deterministic(1),
            ciw.dists.Deterministic(3),
            ciw.dists.Deterministic(7)
        ],
        action_chooser=bedmoves.EpsilonHard(epsilon=0.0, QLearning=None),
        isolation_penalty=2.0,
        adjacent_move_penalty=2.0,
        nonadjacent_move_penalty=2.0,
        QLearning=bedmoves.QLearning(0.5, 0.9, 1.0),
        seed=0
    )
    S.state = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )

    assert S.overall_cost == 0
    S.inflict_cost(11)
    assert S.overall_cost == 99.0

    
    # Resource use 6 per time unit, penalty 0:
    S = bedmoves.BedMoveSimulation(
        arrival_distributions=[
            ciw.dists.Deterministic(5),
            ciw.dists.Deterministic(9),
            ciw.dists.Deterministic(11)
        ],
        los_distributions=[
            ciw.dists.Deterministic(1),
            ciw.dists.Deterministic(3),
            ciw.dists.Deterministic(7)
        ],
        action_chooser=bedmoves.EpsilonHard(epsilon=0.0, QLearning=None),
        isolation_penalty=2.0,
        adjacent_move_penalty=2.0,
        nonadjacent_move_penalty=2.0,
        QLearning=bedmoves.QLearning(0.5, 0.9, 1.0),
        seed=0
    )
    S.state = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 1)
        )
    )

    assert S.overall_cost == 0
    S.inflict_cost(2)
    assert S.overall_cost == 12.0


def test_BedMovesSimulation_arrival_and_exit():
    S = bedmoves.BedMoveSimulation(
        arrival_distributions=[
            ciw.dists.Deterministic(5),
            ciw.dists.Deterministic(9),
            ciw.dists.Deterministic(11)
        ],
        los_distributions=[
            ciw.dists.Deterministic(1),
            ciw.dists.Deterministic(3),
            ciw.dists.Deterministic(7)
        ],
        action_chooser=bedmoves.EpsilonHard(epsilon=0.0, QLearning=None),
        isolation_penalty=2.0,
        adjacent_move_penalty=2.0,
        nonadjacent_move_penalty=2.0,
        QLearning=bedmoves.QLearning(0.5, 0.9, 1.0),
        seed=0
    )
    expected_state_before = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    expected_state_after= np.array(
        (
            (0, 0, 0, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    assert S.next_arrivals == {0: 5, 1: 9, 2: 11}
    assert len(S.patients.patient_types) == 17
    assert len(S.patients.los) == 17
    assert len(S.patients.exit_dates) == 17
    assert len(S.patients.blocks) == 17
    assert np.min(S.patients.patient_types) == -1
    assert np.max(S.patients.patient_types) == -1
    assert np.min(S.patients.blocks) == -1
    assert np.max(S.patients.blocks) == -1
    assert np.min(S.patients.los) == np.inf
    assert np.max(S.patients.los) == np.inf
    assert np.min(S.patients.exit_dates) == np.inf
    assert np.max(S.patients.exit_dates) == np.inf
    assert S.now == 0.0
    assert np.array_equal(S.state, expected_state_before)
    assert np.array_equal(S.patients.free_indices, [i for i in range(17)])

    S.arrival(5, 0)

    assert S.next_arrivals == {0: 10, 1: 9, 2: 11}
    assert S.now == 5.0
    assert np.array_equal(S.state, expected_state_after)
    assert len(S.patients.patient_types) == 17
    assert len(S.patients.los) == 17
    assert len(S.patients.exit_dates) == 17
    assert len(S.patients.blocks) == 17
    assert np.min(S.patients.patient_types) == -1
    assert np.max(S.patients.patient_types) == 0
    assert np.min(S.patients.blocks) == -1
    assert np.max(S.patients.blocks) == 6
    assert np.min(S.patients.los) == 1.0
    assert np.max(S.patients.los) == np.inf
    assert np.min(S.patients.exit_dates) == 6.0
    assert np.max(S.patients.exit_dates) == np.inf
    assert np.array_equal(S.patients.free_indices, [i for i in range(16)])

    S.exit(16)

    assert S.next_arrivals == {0: 10, 1: 9, 2: 11}
    assert len(S.patients.patient_types) == 17
    assert len(S.patients.los) == 17
    assert len(S.patients.exit_dates) == 17
    assert len(S.patients.blocks) == 17
    assert np.min(S.patients.patient_types) == -1
    assert np.max(S.patients.patient_types) == -1
    assert np.min(S.patients.blocks) == -1
    assert np.max(S.patients.blocks) == -1
    assert np.min(S.patients.los) == np.inf
    assert np.max(S.patients.los) == np.inf
    assert np.min(S.patients.exit_dates) == np.inf
    assert np.max(S.patients.exit_dates) == np.inf
    assert S.now == 6.0
    assert np.array_equal(S.state, expected_state_before)


def test_can_simulate_with_initial_Qvals():
    # First test on a state-action I will encounter
    state = (
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ), 2
    )
    action = 7
    initial_qvals = np.array(
        [(9000000000027, 2.5, 34)],
        dtype=[('Key', '<i8'), ('Q', '<f8'), ('Hits', 'i4')]
    )
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=0.2,
        initial_Qvalues=initial_qvals
    )
    S = bedmoves.BedMoveSimulation(
        arrival_distributions=[
            ciw.dists.Exponential(1.5),
            ciw.dists.Exponential(1.0),
            ciw.dists.Exponential(0.5)
        ],
        los_distributions=[
            ciw.dists.Exponential(0.1),
            ciw.dists.Exponential(0.5),
            ciw.dists.Exponential(0.2)
        ],
        action_chooser=bedmoves.EpsilonHard(epsilon=0.0, QLearning=None),
        isolation_penalty=3,
        adjacent_move_penalty=1,
        nonadjacent_move_penalty=2,
        QLearning=Q,
        seed=0
    )
    S.simulate_until_max_time(2)
    initial_state_action_pair = 9000000000027
    assert initial_state_action_pair in Q.Qvals

    # Now repeat for an action I won't encounter
    state = (
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ), 2
    )
    action = 2
    initial_qvals = np.array(
        [(9000000000022, 2.5, 34)],
        dtype=[('Key', '<i8'), ('Q', '<f8'), ('Hits', 'i4')]
    )
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=0.2,
        initial_Qvalues=initial_qvals
    )
    S = bedmoves.BedMoveSimulation(
        arrival_distributions=[
            ciw.dists.Exponential(1.5),
            ciw.dists.Exponential(1.0),
            ciw.dists.Exponential(0.5)
        ],
        los_distributions=[
            ciw.dists.Exponential(0.1),
            ciw.dists.Exponential(0.5),
            ciw.dists.Exponential(0.2)
        ],
        action_chooser=bedmoves.EpsilonHard(epsilon=0.0, QLearning=None),
        isolation_penalty=3,
        adjacent_move_penalty=1,
        nonadjacent_move_penalty=2,
        QLearning=Q,
        seed=0
    )
    S.simulate_until_max_time(2)
    initial_state_action_pair = 9000000000022
    assert initial_state_action_pair in Q.Qvals

    # Now repeat for a state I won't encounter
    state = (
        (
            (1, 1, 1, 1, 1, 1, 1, 1, 1),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ), 2
    )
    action = 2
    initial_qvals = np.array(
        [(9111110000027, 2.5, 34)],
        dtype=[('Key', '<i8'), ('Q', '<f8'), ('Hits', 'i4')]
    )
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=0.2,
        initial_Qvalues=initial_qvals
    )
    S = bedmoves.BedMoveSimulation(
        arrival_distributions=[
            ciw.dists.Exponential(1.5),
            ciw.dists.Exponential(1.0),
            ciw.dists.Exponential(0.5)
        ],
        los_distributions=[
            ciw.dists.Exponential(0.1),
            ciw.dists.Exponential(0.5),
            ciw.dists.Exponential(0.2)
        ],
        action_chooser=bedmoves.EpsilonHard(epsilon=0.0, QLearning=None),
        isolation_penalty=3,
        adjacent_move_penalty=1,
        nonadjacent_move_penalty=2,
        QLearning=Q,
        seed=0
    )
    S.simulate_until_max_time(2)
    initial_state_action_pair = 9111110000027
    assert initial_state_action_pair in Q.Qvals
