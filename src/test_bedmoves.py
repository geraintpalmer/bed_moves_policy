import bedmoves
import pytest
import numpy as np
import ciw
import math
import pandas as pd

class FakeSimulation:
    pass


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
    newS = bedmoves.move_patient(S, 0, 2, 3)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 1, 1, 0, 0)
        )
    )
    newS = bedmoves.move_patient(S, 2, 8, 1)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 1, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    newS = bedmoves.move_patient(S, 1, 4, 2)
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
    newS = bedmoves.insert_patient(S, 0, 2)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 1)
        )
    )
    newS = bedmoves.insert_patient(S, 2, 9)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 2, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    newS = bedmoves.insert_patient(S, 1, 3)
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
    newS = bedmoves.remove_patient(S, 0, 2)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 0, 0)
        )
    )
    newS = bedmoves.remove_patient(S, 2, 8)
    assert np.array_equal(newS, expected_newS)

    expected_newS = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 1, 1, 0)
        )
    )
    newS = bedmoves.remove_patient(S, 1, 4)
    assert np.array_equal(newS, expected_newS)


def test_get_available_insert_moves():
    S = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    )
    expected_moves = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    available_moves = bedmoves.get_available_insert_moves(S)
    assert expected_moves == available_moves

    S = np.array(
        (
            (2, 1, 2, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 0)
        )
    )
    expected_moves = [2, 4, 5, 6, 9]
    available_moves = bedmoves.get_available_insert_moves(S)
    assert expected_moves == available_moves



def test_get_available_moves():
    S = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 1, 1, 0)
        )
    )
    expected_moves = [
        (0, 2, 1), (0, 2, 3), (0, 2, 5), (0, 2, 6), (0, 2, 9),
        (0, 4, 1), (0, 4, 3), (0, 4, 5), (0, 4, 6), (0, 4, 9),
        (1, 1, 3), (1, 1, 5), (1, 1, 6), (1, 1, 9),
        (1, 3, 1), (1, 3, 5), (1, 3, 6), (1, 3, 9),
        (2, 4, 1), (2, 4, 3), (2, 4, 5), (2, 4, 6), (2, 4, 9),
        (2, 6, 1), (2, 6, 3), (2, 6, 5), (2, 6, 9),
        (2, 7, 1), (2, 7, 3), (2, 7, 5), (2, 7, 6), (2, 7, 9),
        (2, 8, 1), (2, 8, 3), (2, 8, 5), (2, 8, 6), (2, 8, 9)
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


def test_combine_dfs():
    A1 = pd.DataFrame({
        'Q': [150.0, 200.0, 300.0],
        'hits': [1, 2, 3]
    }, index=['(111)-1', '(111)-2', '(222)-1'])

    A2 = pd.DataFrame({
        'Q': [50.0, 250.0, 500.0, 100.0],
        'hits': [1, 2, 1, 4]
    }, index=['(111)-1', '(111)-3', '(222)-1', '(333)-2'])

    A = bedmoves.combine_Qvalues([A1, A2])
    expectedA = pd.DataFrame({
        'Q': [100.0, 200.0, 250.0, 350.0, 100.0],
        'hits': [2, 2, 2, 4, 4]
    }, index=['(111)-1', '(111)-2', '(111)-3', '(222)-1', '(333)-2'])
    pd.testing.assert_frame_equal(A, expectedA)


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
    assert RC.choose_arriving_block(S9, 0) == 9
    assert RC.choose_arriving_block(S9, 0) == 9
    assert RC.choose_arriving_block(S9, 0) == 9
    assert RC.choose_arriving_block(S9, 0) == 9
    assert RC.choose_arriving_block(S9, 0) == 9
    assert RC.choose_arriving_block(S9, 0) == 9

    ciw.seed(0)
    assert RC.choose_arriving_block(S123, 0) in [1, 2, 3]
    assert RC.choose_arriving_block(S123, 0) in [1, 2, 3]
    assert RC.choose_arriving_block(S123, 0) in [1, 2, 3]
    assert RC.choose_arriving_block(S123, 0) in [1, 2, 3]
    assert RC.choose_arriving_block(S123, 0) in [1, 2, 3]
    assert RC.choose_arriving_block(S123, 0) in [1, 2, 3]

    ciw.seed(0)
    assert RC.choose_arriving_block(Sfull, 0) == False
    assert RC.choose_arriving_block(Sfull, 0) == False
    assert RC.choose_arriving_block(Sfull, 0) == False
    assert RC.choose_arriving_block(Sfull, 0) == False
    assert RC.choose_arriving_block(Sfull, 0) == False
    assert RC.choose_arriving_block(Sfull, 0) == False


def test_epsilonhard_10():
    S123 = np.array(
        (
            (2, 1, 1, 3, 2, 2, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 1)
        )
    )
    hashS123_1 = bedmoves.QLearning.get_hash_state(None, (S123, 0), 1)
    hashS123_2 = bedmoves.QLearning.get_hash_state(None, (S123, 0), 2)
    hashS123_3 = bedmoves.QLearning.get_hash_state(None, (S123, 0), 3)
    Qdf = pd.DataFrame(
        {
            'Q': [0.35, 1.56, 0.98],
            'hits': [34, 12, 55]
        },
        index=[hashS123_1, hashS123_2, hashS123_3]
    )
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=2.0,
        initial_Qvalues=Qdf
    )
    EH = bedmoves.EpsilonHard(epsilon=1.0, QLearning=Q)
    ciw.seed(0)
    assert EH.choose_arriving_block(S123, 0) == 2
    assert EH.choose_arriving_block(S123, 0) == 2
    assert EH.choose_arriving_block(S123, 0) == 2
    assert EH.choose_arriving_block(S123, 0) == 2
    assert EH.choose_arriving_block(S123, 0) == 2
    assert EH.choose_arriving_block(S123, 0) == 2


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
    hashS123_1 = bedmoves.QLearning.get_hash_state(None, (S123, 0), 1)
    hashS123_2 = bedmoves.QLearning.get_hash_state(None, (S123, 0), 2)
    hashS123_3 = bedmoves.QLearning.get_hash_state(None, (S123, 0), 3)
    Qdf = pd.DataFrame(
        {
            'Q': [0.35, 1.56, 0.98],
            'hits': [34, 12, 55]
        },
        index=[hashS123_1, hashS123_2, hashS123_3]
    )
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=2.0,
        initial_Qvalues=Qdf
    )
    EH = bedmoves.EpsilonHard(epsilon=0.7, QLearning=Q)
    ciw.seed(0)
    choices = [EH.choose_arriving_block(S123, 0) for _ in range(1000)]
    assert sum(c == 2 for c in choices) == 781



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
    assert len(Q.Qvals_df) == 0
    assert Q.hash_state == None

    FS = FakeSimulation()
    Q.attach_simulation(FS)
    assert Q.simulation == FS

def test_QLearning_initial_Qvalues():
    S1 = (
            (
            (1, 1, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ), 0
    )
    S2 = (
            (
            (1, 1, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ), 1
    )
    S3 = (
            (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 1, 1, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ), 0
    )

    Qdf = pd.DataFrame(
        {
            'Q': [1.2, 1.8, 1.1, 0.7, 7.1, 3.1],
            'hits': [34, 12, 55, 2, 5, 6]
        }, index=[str(S1) + '-4', str(S1) + '-5', str(S2) + '-4', str(S2) + '-5', str(S2) + '-6', str(S3) + '-2']
    )

    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=1.0,
        initial_Qvalues=Qdf
        )

    assert Q.learning_rate == 0.5
    assert Q.discount_factor == 0.9
    assert Q.transform_parameter == 1.0
    assert Q.previous_cost == 0.0
    assert Q.hash_state == None

    assert len(Q.Qvals_df) == 6
    assert np.array_equal(Q.Qvals_df.index, [str(S1) + '-4', str(S1) + '-5', str(S2) + '-4', str(S2) + '-5', str(S2) + '-6', str(S3) + '-2'])
    assert np.array_equal(Q.Qvals_df['Q'], [1.2, 1.8, 1.1, 0.7, 7.1, 3.1])
    assert np.array_equal(Q.Qvals_df['hits'], [34, 12, 55, 2, 5, 6])

    # Test that changing Q values do not affect original df
    Q.Qvals_df.loc[str(S1) + '-4', 'Q'] = 500.7
    Q.Qvals_df.loc[str(S1) + '-4', 'hits'] += 1
    assert Q.Qvals_df.loc[str(S1) + '-4', 'Q'] == 500.7
    assert Q.Qvals_df.loc[str(S1) + '-4', 'hits'] == 35
    assert Qdf.loc[str(S1) + '-4', 'Q'] == 1.2
    assert Qdf.loc[str(S1) + '-4', 'hits'] == 34


def test_QLearning_hashstate():
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=1.0
        )
    S = (
        np.array(
            (
                (0, 2, 0, 2, 0, 0, 0, 0, 0),
                (1, 0, 1, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 1, 0, 1, 1, 1, 0)
            )
        ),
        1
    )
    assert Q.get_hash_state(S, 2) == str((
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 1, 1, 0)
        ), 1
    )) + '-2'


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
    S = (np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    ), 0)
    Q.hash_state = Q.get_hash_state(S, 1)
    Q.previous_cost = 90

    FS = FakeSimulation()
    FS.overall_cost = 100
    FS.now = 40
    Q.attach_simulation(FS)

    next_state = (np.array(
        (
            (1, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    ), 2)
    next_action = 5

    Q.update_Q_values(next_state, next_action)

    assert Q.hash_state == Q.get_hash_state(next_state, 5)
    R = math.exp(-Q.transform_parameter * 10)
    assert Q.Qvals_dict[Q.get_hash_state(S, 1)] == 0.5 * R
    assert Q.Qhits_dict[Q.get_hash_state(S, 1)] == 1


def test_Patient_class():
    P = bedmoves.Patient(
        patient_type=0,
        los=12.3,
        arrival_date=4.8,
        block=4
    )
    assert P.patient_type == 0
    assert P.los == 12.3
    assert P.arrival_date == 4.8
    assert P.exit_date == 17.1
    assert P.block == 4

    P = bedmoves.Patient(
        patient_type=1,
        los=0.3674,
        arrival_date=674.2119,
        block=9
    )
    assert P.patient_type == 1
    assert P.los == 0.3674
    assert P.arrival_date == 674.2119
    assert P.exit_date == 674.5793
    assert P.block == 9


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
        Qlearning=bedmoves.QLearning(0.5, 0.9, 1.0),
        seed=0
    )

    assert len(S.arrival_distributions) == 3
    assert len(S.los_distributions) == 3
    assert str(S.action_chooser) == "EpsilonHard-0.0"
    assert S.next_arrivals == {0: 5, 1: 9, 2: 11}
    assert str(S.Qlearning) == "QLearning"
    assert S.prev_now == 0.0
    assert S.now == 0.0
    assert S.overall_cost == 0.0
    assert S.isolation_penalty == 2.0
    assert S.adjacent_move_penalty == 2.0
    assert S.nonadjacent_move_penalty == 2.0
    assert len(S.patients) == 0


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
        Qlearning=bedmoves.QLearning(0.5, 0.9, 1.0),
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
        Qlearning=bedmoves.QLearning(0.5, 0.9, 1.0),
        seed=0
    )

    next_exit, next_patient = S.find_next_exit_date()
    assert next_exit == float('inf')
    assert next_patient == None

    S.patients = [
        bedmoves.Patient(patient_type=0, los=31.5, arrival_date=2.4, block=1),
        bedmoves.Patient(patient_type=0, los=5.6, arrival_date=6.1, block=1),
        bedmoves.Patient(patient_type=2, los=22.2, arrival_date=1.2, block=7),
        bedmoves.Patient(patient_type=1, los=9.6, arrival_date=7.3, block=4),
    ]

    next_exit, next_patient = S.find_next_exit_date()
    assert next_exit == 11.7
    assert next_patient.patient_type == 0
    assert next_patient.los == 5.6
    assert next_patient.arrival_date == 6.1
    assert next_patient.exit_date == 11.7
    assert next_patient.block == 1


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
        Qlearning=bedmoves.QLearning(0.5, 0.9, 1.0),
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
        Qlearning=bedmoves.QLearning(0.5, 0.9, 1.0),
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
        Qlearning=bedmoves.QLearning(0.5, 0.9, 1.0),
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
    assert len(S.patients) == 0
    assert S.now == 0.0
    assert np.array_equal(S.state, expected_state_before)

    S.arrival(5, 0)

    assert S.next_arrivals == {0: 10, 1: 9, 2: 11}
    assert len(S.patients) == 1
    assert S.now == 5.0
    assert np.array_equal(S.state, expected_state_after)
    assert S.patients[0].patient_type == 0
    assert S.patients[0].arrival_date == 5.0
    assert S.patients[0].los == 1.0
    assert S.patients[0].exit_date == 6.0

    S.exit(S.patients[0])

    assert S.next_arrivals == {0: 10, 1: 9, 2: 11}
    assert len(S.patients) == 0
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
    Qdf = pd.DataFrame(
        {
            'Q': [2.5],
            'hits': [34]
        }, index=[str(state) + '-' + str(action)]
    )
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=0.2,
        initial_Qvalues=Qdf
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
        Qlearning=Q,
        seed=0
    )
    S.simulate_until_max_time(2)
    vals = Q.Qvals_df
    state_action_paris = vals.index
    initial_state_action_pair = str(state) + '-' + str(action)
    assert any(state_action_paris == initial_state_action_pair)

    # Now repeat for an action I won't encounter
    state = (
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ), 2
    )
    action = 2
    Qdf = pd.DataFrame(
        {
            'Q': [2.5],
            'hits': [34]
        }, index=[str(state) + '-' + str(action)]
    )
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=0.2,
        initial_Qvalues=Qdf
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
        Qlearning=Q,
        seed=0
    )
    S.simulate_until_max_time(2)
    vals = Q.Qvals_df
    state_action_paris = vals.index
    initial_state_action_pair = str(state) + '-' + str(action)
    assert any(state_action_paris == initial_state_action_pair)

    # Now repeat for an state I won't encounter
    state = (
        (
            (1, 1, 1, 1, 1, 1, 1, 1, 1),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ), 2
    )
    action = 2
    Qdf = pd.DataFrame(
        {
            'Q': [2.5],
            'hits': [34]
        }, index=[str(state) + '-' + str(action)]
    )
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=0.2,
        initial_Qvalues=Qdf
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
        Qlearning=Q,
        seed=0
    )
    S.simulate_until_max_time(2)
    vals = Q.Qvals_df
    state_action_paris = vals.index
    initial_state_action_pair = str(state) + '-' + str(action)
    assert any(state_action_paris == initial_state_action_pair)
