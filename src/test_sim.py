import sim
import pytest
import numpy as np
import ciw

def test_find_next_arrival_date():
    next_arrivals = np.array([56.7, 12.2, 34.3])
    t, p = sim.find_next_arrival_date(next_arrivals)
    assert t == 12.2
    assert p == 1

    next_arrivals = np.array([6.7, 12.2, 34.3])
    t, p = sim.find_next_arrival_date(next_arrivals)
    assert t == 6.7
    assert p == 0

    next_arrivals = np.array([56.7, 182.2, 34.3])
    t, p = sim.find_next_arrival_date(next_arrivals)
    assert t == 34.3
    assert p == 2

    next_arrivals = np.array([0.442, 0.432, 0.478])
    t, p = sim.find_next_arrival_date(next_arrivals)
    assert t == 0.432
    assert p == 1

def test_find_next_exit_date():
    exit_dates = np.array(
        [17.4, 10.5, 34.6, 9.1, 13.9, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    )
    t, i =  sim.find_next_exit_date(exit_dates)
    assert t == 9.1
    assert i == 3

    exit_dates = np.array(
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 17.4, 10.5, 34.6, 9.1, 13.9]
    )
    t, i =  sim.find_next_exit_date(exit_dates)
    assert t == 9.1
    assert i == 15

    exit_dates = np.array(
        [np.inf, np.inf, np.inf, np.inf, np.inf, 17.4, 1.5, 34.6, 9.1, 13.9, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    )
    t, i =  sim.find_next_exit_date(exit_dates)
    assert t == 1.5
    assert i == 6

    exit_dates = np.array(
        [17.4, 10.5, 34.6, 9.1, 13.9, np.inf, np.inf, np.inf, np.inf, np.inf, 44.2, 6.11, np.inf, np.inf, np.inf, np.inf, np.inf]
    )
    t, i =  sim.find_next_exit_date(exit_dates)
    assert t == 6.11
    assert i == 11


def test_get_cost():
    S = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0)
    )
    cost = sim.get_cost(
        state=S,
        update_time=11,
        prev_time=0,
        isolation_penalty=2.0
    )
    assert cost == 99.0

    S = np.array(
        (0, 2, 0, 2, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 1, 1)
    )

    cost = sim.get_cost(
        state=S,
        update_time=28.3,
        prev_time=26.3,
        isolation_penalty=2.0
    )
    assert cost == 12.0


def test_WardRLSimulation_arrival_and_exit():
    S = sim.WardRLSimulation(
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
        isolation_penalty=2.0,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=1.0,
        seed=0
    )
    expected_state_before = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_state_after = np.array(
        (0, 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert np.array_equal(S.next_arrivals, np.array([5.0, 9.0, 11.0]))
    assert len(S.patients_patient_types) == 17
    assert len(S.patients_exit_dates) == 17
    assert len(S.patients_blocks) == 17
    assert len(S.patients_free_indices) == 17
    assert S.patients_number_free == 17
    assert np.min(S.patients_patient_types) == -1
    assert np.max(S.patients_patient_types) == -1
    assert np.min(S.patients_blocks) == -1
    assert np.max(S.patients_blocks) == -1
    assert np.min(S.patients_exit_dates) == np.inf
    assert np.max(S.patients_exit_dates) == np.inf
    assert S.now == 0.0
    assert np.array_equal(S.state, expected_state_before)
    assert np.array_equal(S.patients_free_indices, [i for i in range(17)])

    S.arrival(next_arrival=5, patient_type=0)

    assert np.array_equal(S.next_arrivals, np.array([10.0, 9.0, 11.0]))
    assert S.now == 5.0
    assert np.array_equal(S.state, expected_state_after)
    assert len(S.patients_patient_types) == 17
    assert len(S.patients_exit_dates) == 17
    assert len(S.patients_blocks) == 17
    assert len(S.patients_free_indices) == 16
    assert np.min(S.patients_patient_types) == -1
    assert np.max(S.patients_patient_types) == 0
    assert np.min(S.patients_blocks) == -1
    assert np.max(S.patients_blocks) == 5
    assert np.min(S.patients_exit_dates) == 6.0
    assert np.max(S.patients_exit_dates) == np.inf
    assert np.array_equal(S.patients_free_indices, [i for i in range(16)])

    S.exit(patient_idx=16)

    assert np.array_equal(S.next_arrivals, np.array([10.0, 9.0, 11.0]))
    assert len(S.patients_patient_types) == 17
    assert len(S.patients_exit_dates) == 17
    assert len(S.patients_blocks) == 17
    assert len(S.patients_free_indices) == 17
    assert np.min(S.patients_patient_types) == -1
    assert np.max(S.patients_patient_types) == -1
    assert np.min(S.patients_blocks) == -1
    assert np.max(S.patients_blocks) == -1
    assert np.min(S.patients_exit_dates) == np.inf
    assert np.max(S.patients_exit_dates) == np.inf
    assert S.now == 6.0
    assert np.array_equal(S.state, expected_state_before)
    assert np.array_equal(S.patients_free_indices, [i for i in range(17)])


def test_can_simulate_with_initial_Qvals():
    # First test on a state-action I will encounter
    keys = np.array([100000])
    qval = np.array([2.5])
    hits = np.array([34])
    
    S = sim.WardRLSimulation(
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
        isolation_penalty=3,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=0.2,
        seed=0,
        initial_Qvalues=(keys, qval),
        learn=True
    )
    S.simulate_until_max_time(2)
    assert 100003 in S.Qvals
    assert 22 not in S.Qvals
    assert 162521625229227 not in S.Qvals
    assert 100003 in S.hits
    assert 22 not in S.hits
    assert 162521625229227 not in S.hits

    # Now repeat for an action I won't encounter
    keys = np.array([22])
    qval = np.array([2.5])
    hits = np.array([34])
    
    S = sim.WardRLSimulation(
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
        isolation_penalty=3,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=0.2,
        seed=0,
        initial_Qvalues=(keys, qval),
        learn=True
    )
    S.simulate_until_max_time(2)
    assert 100003 in S.Qvals
    assert 22 in S.Qvals
    assert 162521625229227 not in S.Qvals
    assert 100003 in S.hits
    assert 22 in S.hits
    assert 162521625229227 not in S.hits

    # Now repeat for a state I won't encounter
    keys = np.array([162521625229227])
    qval = np.array([2.5])
    hits = np.array([34])
    
    S = sim.WardRLSimulation(
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
        isolation_penalty=3,
        epsilon=0.0,
        learning_rate=0.5,
        discount_factor=0.9,
        transform_parameter=0.2,
        seed=0,
        initial_Qvalues=(keys, qval),
        learn=True
    )
    S.simulate_until_max_time(2)
    assert 100003 in S.Qvals
    assert 22 not in S.Qvals
    assert 162521625229227 in S.Qvals
    assert 100003 in S.hits
    assert 22 not in S.hits
    assert 162521625229227 in S.hits
