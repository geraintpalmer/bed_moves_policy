import bedmoves
import pytest
import numpy as np
import ciw
import math

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


def test_random_choice_chooser():
    RC = bedmoves.RandomChoice()
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


def test_Agent():
    S = np.array(
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 1, 1, 0)
        )
    )

    A1 = bedmoves.Agent(state=S, action=1, Q=0.0)
    A2 = bedmoves.Agent(state=S, action=9, Q=3.5)

    assert np.array_equal(A1.state, S)
    assert np.array_equal(A2.state, S)
    assert A1.action == 1
    assert A2.action == 9
    assert A1.Q == 0.0
    assert A2.Q == 3.5
    assert np.array_equal(A1.Qts, [0.0])
    assert np.array_equal(A2.Qts, [3.5])
    assert np.array_equal(A1.ts, [0.0])
    assert np.array_equal(A2.ts, [0.0])


    A1.update_Q(newQ=1.5, t=0.8)
    A2.update_Q(newQ=2.9, t=1.1)

    assert np.array_equal(A1.state, S)
    assert np.array_equal(A2.state, S)
    assert A1.action == 1
    assert A2.action == 9
    assert A1.Q == 1.5
    assert A2.Q == 2.9
    assert np.array_equal(A1.Qts, [0.0, 1.5])
    assert np.array_equal(A2.Qts, [3.5, 2.9])
    assert np.array_equal(A1.ts, [0.0, 0.8])
    assert np.array_equal(A2.ts, [0.0, 1.1])


def test_QLearning_init():
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_rate=0.9,
        transform_parameter=1.0
        )

    assert Q.learning_rate == 0.5
    assert Q.discount_rate == 0.9
    assert Q.transform_parameter == 1.0
    assert Q.previous_cost == 0.0
    assert Q.agents == {}
    assert Q.state == None
    assert Q.action == None

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

    initial_Qvalues = {
        S1: {4: 1.2, 5: 1.8},
        S2: {4: 1.1, 5: 0.7, 6: 7.1},
        S3: {2: 3.1}
    }
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_rate=0.9,
        transform_parameter=1.0,
        initial_Qvalues=initial_Qvalues
        )

    assert Q.learning_rate == 0.5
    assert Q.discount_rate == 0.9
    assert Q.transform_parameter == 1.0
    assert Q.previous_cost == 0.0
    assert Q.state == None
    assert Q.action == None

    assert len(Q.agents.keys()) == 3 
    assert S1 in Q.agents
    assert S2 in Q.agents
    assert S3 in Q.agents
    assert len(Q.agents[S1]) == 2
    assert len(Q.agents[S2]) == 3
    assert len(Q.agents[S3]) == 1
    assert 4 in Q.agents[S1]
    assert 5 in Q.agents[S1]
    assert 4 in Q.agents[S2]
    assert 5 in Q.agents[S2]
    assert 6 in Q.agents[S2]
    assert 2 in Q.agents[S3]
    assert Q.agents[S1][4].Q == 1.2
    assert Q.agents[S1][5].Q == 1.8
    assert Q.agents[S2][4].Q == 1.1
    assert Q.agents[S2][5].Q == 0.7
    assert Q.agents[S2][6].Q == 7.1
    assert Q.agents[S3][2].Q == 3.1


def test_QLearning_hashstate():
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_rate=0.9,
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
    assert Q.get_hash_state(S) == (
        (
            (0, 2, 0, 2, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 1, 1, 0)
        ), 1
    )


def test_transform_cost():
    Q = bedmoves.QLearning(
        learning_rate=0.5,
        discount_rate=0.9,
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
        discount_rate=0.9,
        transform_parameter=1.0
        )
    S = (np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
    ), 0)
    Q.state = S
    Q.hash_state = Q.get_hash_state(Q.state)
    Q.action = 1
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

    assert np.array_equal(Q.state[0], next_state[0])
    assert Q.state[1], next_state[1]
    assert Q.action == next_action

    agent = Q.agents[Q.get_hash_state(S)][1]
    R = math.exp(-Q.transform_parameter * 10)
    assert agent.Q == 0.5 * R
    assert len(agent.Qts) == 2
    assert np.array_equal(agent.ts, [0.0, 40])


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
        action_chooser=bedmoves.RandomChoice(),
        isolation_penalty=2.0,
        adjacent_move_penalty=2.0,
        nonadjacent_move_penalty=2.0,
        Qlearning=bedmoves.QLearning(0.5, 0.9, 1.0),
        seed=0
    )

    assert len(S.arrival_distributions) == 3
    assert len(S.los_distributions) == 3
    assert str(S.action_chooser) == "RandomChoice"
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
        action_chooser=bedmoves.RandomChoice(),
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
        action_chooser=bedmoves.RandomChoice(),
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
        action_chooser=bedmoves.RandomChoice(),
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
        action_chooser=bedmoves.RandomChoice(),
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
        action_chooser=bedmoves.RandomChoice(),
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

