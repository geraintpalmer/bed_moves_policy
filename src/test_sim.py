import sim
import pytest
import numpy as np

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

def test_find_next_activity_date():
    exit_dates = np.array(
        [17.4, 10.5, 34.6, 9.1, 13.9, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    )
    t, i =  sim.find_next_activity_date(exit_dates)
    assert t == 9.1
    assert i == 3

    exit_dates = np.array(
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 17.4, 10.5, 34.6, 9.1, 13.9]
    )
    t, i =  sim.find_next_activity_date(exit_dates)
    assert t == 9.1
    assert i == 15

    exit_dates = np.array(
        [np.inf, np.inf, np.inf, np.inf, np.inf, 17.4, 1.5, 34.6, 9.1, 13.9, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    )
    t, i =  sim.find_next_activity_date(exit_dates)
    assert t == 1.5
    assert i == 6

    exit_dates = np.array(
        [17.4, 10.5, 34.6, 9.1, 13.9, np.inf, np.inf, np.inf, np.inf, np.inf, 44.2, 6.11, np.inf, np.inf, np.inf, np.inf, np.inf]
    )
    t, i =  sim.find_next_activity_date(exit_dates)
    assert t == 6.11
    assert i == 11


def test_get_state_cost():
    S = np.array(
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    cost = sim.get_state_cost(
        state=S,
        update_time=11,
        prev_time=0,
        isolation_penalty=2.0
    )
    assert cost == 99.0

    S = np.array(
        (0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2)
    )

    cost = sim.get_state_cost(
        state=S,
        update_time=28.3,
        prev_time=26.3,
        isolation_penalty=2.0
    )
    assert cost == 12.0


def test_WardSimulation_arrival_and_exit():
    S = sim.WardTraining(
        arrival_distributions=[
            ('Deterministic', 5),
            ('Deterministic', 9),
            ('Deterministic', 11)
        ],
        los_distributions=[
            ('Deterministic', 1),
            ('Deterministic', 3),
            ('Deterministic', 7)
        ],
        deterioration_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=2.0,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=500.0,
        learning_rate=0.5,
        discount_factor=0.9
    )
    expected_state_before = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_state_after = np.array(
        (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
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
    assert S.max_time == 500.0
    assert np.array_equal(S.state, expected_state_before)
    assert np.array_equal(S.patients_free_indices, [i for i in range(17)])

    S.arrival(next_arrival=5.0, patient_type=0)

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
    keys = np.array([93388800101], dtype=np.int64)
    qval = np.array([2.5], dtype=np.float32)
    hits = np.array([34], dtype=np.int16)
    
    S = sim.WardTraining(
        arrival_distributions=[
            ('Exponential', 1.5),
            ('Exponential', 1.0),
            ('Exponential', 0.5)
        ],
        los_distributions=[
            ('Exponential', 0.1),
            ('Exponential', 0.5),
            ('Exponential', 0.2)
        ],
        deterioration_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=2.0,
        learning_rate=0.5,
        discount_factor=0.9,
        initial_keys=keys,
        initial_qvals=qval
    )
    S.simulate_until_max_time()
    assert np.int64(93388800101) in S.Q_index_map
    assert np.int64(22) not in S.Q_index_map
    assert np.int64(162521625229227) not in S.Q_index_map

    # Now repeat for an action I won't encounter
    keys = np.array([22], dtype=np.int64)
    qval = np.array([2.5], dtype=np.float32)
    hits = np.array([34], dtype=np.int16)
    
    S = sim.WardTraining(
        arrival_distributions=[
            ('Exponential', 1.5),
            ('Exponential', 1.0),
            ('Exponential', 0.5)
        ],
        los_distributions=[
            ('Exponential', 0.1),
            ('Exponential', 0.5),
            ('Exponential', 0.2)
        ],
        deterioration_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=2.0,
        learning_rate=0.5,
        discount_factor=0.9,
        initial_keys=keys,
        initial_qvals=qval
    )
    S.simulate_until_max_time()
    assert np.int64(93388800101) in S.Q_index_map
    assert np.int64(22) in S.Q_index_map
    assert np.int64(162521625229227) not in S.Q_index_map

    # Now repeat for a state I won't encounter
    keys = np.array([162521625229227], dtype=np.int64)
    qval = np.array([2.5], dtype=np.float32)
    hits = np.array([34], dtype=np.int16)
    
    S = sim.WardTraining(
        arrival_distributions=[
            ('Exponential', 1.5),
            ('Exponential', 1.0),
            ('Exponential', 0.5)
        ],
        los_distributions=[
            ('Exponential', 0.1),
            ('Exponential', 0.5),
            ('Exponential', 0.2)
        ],
        deterioration_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=2.0,
        learning_rate=0.5,
        discount_factor=0.9,
        initial_keys=keys,
        initial_qvals=qval
    )
    S.simulate_until_max_time()
    assert np.int64(93388800101) in S.Q_index_map
    assert np.int64(22) not in S.Q_index_map
    assert np.int64(162521625229227) in S.Q_index_map

def test_using_warmup():
    S = sim.WardEvaluation(
        arrival_distributions=[
            ('Exponential', 1.5),
            ('Exponential', 1.0),
            ('Exponential', 0.5)
        ],
        los_distributions=[
            ('Exponential', 0.1),
            ('Exponential', 0.5),
            ('Exponential', 0.2)
        ],
        deterioration_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=40.0,
        warmup=50.0
    )
    # Simulate for less than the warmup time
    S.simulate_until_max_time()
    assert S.overall_cost == 691.60815
    assert S.warmup_cost == 691.60815

    S = sim.WardEvaluation(
        arrival_distributions=[
            ('Exponential', 1.5),
            ('Exponential', 1.0),
            ('Exponential', 0.5)
        ],
        los_distributions=[
            ('Exponential', 0.1),
            ('Exponential', 0.5),
            ('Exponential', 0.2)
        ],
        deterioration_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=60.0,
        warmup=50.0
    )
    # Simulate for more than the warmup time
    S.simulate_until_max_time()
    assert S.overall_cost == 1014.19354
    assert S.warmup_cost == 834.87354


def test_deterioration():
    S = sim.WardEvaluation(
        arrival_distributions=[
            ('Deterministic', 7.0),
            ('Deterministic', 13.0),
            ('Deterministic', 22.0)
        ],
        los_distributions=[
            ('Deterministic', 10.0),
            ('Deterministic', 10.0),
            ('Deterministic', 10.0)
        ],
        deterioration_distributions=[
            ('Deterministic', 2.0),
            ('Deterministic', 2.0)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=6.0, # only one arrival
        warmup=50.0,
    )

    S_A = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int32
    )
    S_B = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int32
    )
    
    S.simulate_until_max_time()
    assert S.now == 7.0
    assert np.array_equal(S.state, S_A)

    S = sim.WardEvaluation(
        arrival_distributions=[
            ('Deterministic', 7.0),
            ('Deterministic', 13.0),
            ('Deterministic', 22.0)
        ],
        los_distributions=[
            ('Deterministic', 10.0),
            ('Deterministic', 10.0),
            ('Deterministic', 10.0)
        ],
        deterioration_distributions=[
            ('Deterministic', 2.0),
            ('Deterministic', 2.0)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=8.0, # only one arrival, but deteriorates
        warmup=50.0,
    )
    S.simulate_until_max_time()
    assert S.now == 9.0
    assert np.array_equal(S.state, S_B)


def test_improvement():
    S = sim.WardEvaluation(
        arrival_distributions=[
            ('Deterministic', 22.0),
            ('Deterministic', 13.0),
            ('Deterministic', 7.0)
        ],
        los_distributions=[
            ('Deterministic', 10.0),
            ('Deterministic', 10.0),
            ('Deterministic', 10.0)
        ],
        deterioration_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        improvement_distributions=[
            ('Deterministic', 2.0),
            ('Deterministic', 2.0)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=6.0, # only one arrival
        warmup=50.0,
    )

    S_A = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), dtype=np.int32
    )
    S_B = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int32
    )
    
    S.simulate_until_max_time()
    assert S.now == 7.0
    assert np.array_equal(S.state, S_A)

    S = sim.WardEvaluation(
        arrival_distributions=[
            ('Deterministic', 22.0),
            ('Deterministic', 13.0),
            ('Deterministic', 7.0)
        ],
        los_distributions=[
            ('Deterministic', 10.0),
            ('Deterministic', 10.0),
            ('Deterministic', 10.0)
        ],
        deterioration_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        improvement_distributions=[
            ('Deterministic', 2.0),
            ('Deterministic', 2.0)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=8.0, # only one arrival, but deteriorates
        warmup=50.0,
    )
    assert len(S.arrival_distributions) == 3
    assert len(S.los_distributions) == 3
    assert len(S.deterioration_distributions) == 3
    assert len(S.improvement_distributions) == 3
    S.simulate_until_max_time()
    assert S.now == 9.0
    assert np.array_equal(S.state, S_B)


def test_initial_array_preallocations():
    S = sim.WardTraining(
        arrival_distributions=[
            ('Exponential', 0.5),
            ('Exponential', 2.0),
            ('Exponential', 1.5)
        ],
        los_distributions=[
            ('Deterministic', 10.0),
            ('Deterministic', 10.0),
            ('Deterministic', 10.0)
        ],
        deterioration_distributions=[
            ('Deterministic', 2.0),
            ('Deterministic', 2.0)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=100.0,
        warmup=50.0,
    )
    # should initialise to 800
    assert len(S.states) == 800
    assert len(S.Qvals) == 800
    assert len(S.hits) == 800

    S = sim.WardTraining(
        arrival_distributions=[
            ('Exponential', 0.5),
            ('Exponential', 2.0),
            ('Exponential', 1.5)
        ],
        los_distributions=[
            ('Deterministic', 10.0),
            ('Deterministic', 10.0),
            ('Deterministic', 10.0)
        ],
        deterioration_distributions=[
            ('Deterministic', 2.0),
            ('Deterministic', 2.0)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=100.0,
        warmup=50.0,
        M=1357
    )
    # should initialise to 1357
    assert len(S.states) == 1357
    assert len(S.Qvals) == 1357
    assert len(S.hits) == 1357


def test_long_training_run():
    S = sim.WardTraining(
        arrival_distributions=[
            ('Exponential', 0.5),
            ('Exponential', 2.0),
            ('Exponential', 1.5)
        ],
        los_distributions=[
            ('Exponential', 1.0),
            ('Exponential', 1.5),
            ('Exponential', 0.5)
        ],
        deterioration_distributions=[
            ('Exponential', 2.0),
            ('Exponential', 2.0)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.5,
        learning_rate=0.5,
        discount_factor=0.8,
        seed=0,
        max_time=10000.0,
        warmup=1000.0
    )
    S.simulate_until_max_time()

    assert S.overall_cost == 166081.34
    assert S.max_idx == 30917
    assert len(S.Q_index_map) == 30917

def test_long_evaluation_run():
    S = sim.WardEvaluation(
        arrival_distributions=[
            ('Exponential', 0.5),
            ('Exponential', 2.0),
            ('Exponential', 1.5)
        ],
        los_distributions=[
            ('Exponential', 1.0),
            ('Exponential', 1.5),
            ('Exponential', 0.5)
        ],
        deterioration_distributions=[
            ('Exponential', 2.0),
            ('Exponential', 2.0)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=1.0,
        seed=0,
        max_time=10000.0,
        warmup=1000.0
    )
    S.simulate_until_max_time()

    assert S.overall_cost == 164496.78

def test_give_policy():
    S = sim.WardEvaluation(
        arrival_distributions=[
            ('Deterministic', 7.0),
            ('Deterministic', 13.0),
            ('Deterministic', 22.0)
        ],
        los_distributions=[
            ('Deterministic', 10.0),
            ('Deterministic', 10.0),
            ('Deterministic', 10.0)
        ],
        deterioration_distributions=[
            ('Deterministic', 2.0),
            ('Deterministic', 2.0)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        epsilon=0.0,
        seed=0,
        max_time=800.0,
        initial_keys=np.array([11000, 33000, 22000, 44000], dtype=np.int64),
        initial_policy=np.array([404, 101, 101, 505], dtype=np.int16),
        warmup=50.0,
    )

    assert len(S.policy) == 4
    assert S.policy[11000] == 404
    assert S.policy[33000] == 101
    assert S.policy[22000] == 101
    assert S.policy[44000] == 505


def test_state_dependent_arrivals():
    S = sim.WardTraining(
        arrival_distributions=[
            ('Deterministic', 1.0),
            ('Deterministic', 1.1),
            ('Deterministic', 1.2)
        ],
        los_distributions=[
            ('Deterministic', 100.0),
            ('Deterministic', 100.0),
            ('Deterministic', 100.0)
        ],
        deterioration_distributions=[
            ('Deterministic', 100.0),
            ('Deterministic', 100.0)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        learning_rate=0.5,
        discount_factor=0.5,
        epsilon=0.0,
        seed=0,
        max_time=4.9, # I expect 12 arrivals (1, 1.1, 1.2, 2, 2.2, 2.4, 3, 3.3, 3.6, 4, 4.4, 4.8)
        warmup=1.0,
    )
    S.simulate_until_max_time()
    assert len(S.Q_index_map) == 12
    assert sum(S.state) == 13

    S = sim.WardTraining(
        arrival_distributions=[
            ('Deterministic', 1.0),
            ('Deterministic', 1.1),
            ('Deterministic', 1.2)
        ],
        los_distributions=[
            ('Deterministic', 100.0),
            ('Deterministic', 100.0),
            ('Deterministic', 100.0)
        ],
        deterioration_distributions=[
            ('Deterministic', 100.0),
            ('Deterministic', 100.0)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.ones(18)*0.5,
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        learning_rate=0.5,
        discount_factor=0.5,
        epsilon=0.0,
        seed=0,
        max_time=4.9, # I expect 13 arrivals (1, 1.1, 1.2, 2, 2.2, 2.4, 3, 3.3, 3.6, 4, 4.4, 4.8, one at 5.0 to 'end' the loop), but half are discarded.
        warmup=1.0,
    )
    S.simulate_until_max_time()
    assert len(S.Q_index_map) == 3
    assert sum(S.state) == 4

    S = sim.WardTraining(
        arrival_distributions=[
            ('Deterministic', 1.0),
            ('Deterministic', 1.1),
            ('Deterministic', 1.2)
        ],
        los_distributions=[
            ('Deterministic', 100.0),
            ('Deterministic', 100.0),
            ('Deterministic', 100.0)
        ],
        deterioration_distributions=[
            ('Deterministic', 100.0),
            ('Deterministic', 100.0)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        isolation_penalty=3,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=10.0,
        learning_rate=0.5,
        discount_factor=0.5,
        epsilon=0.0,
        seed=0,
        max_time=4.9, # I expect 8 arrivals only (stop arrivals after occupancy 8)
        warmup=1.0,
    )
    S.simulate_until_max_time()
    assert len(S.Q_index_map) == 7
    assert sum(S.state) == 8