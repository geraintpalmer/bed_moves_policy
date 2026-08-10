import ward
import pytest
import numpy as np

def test_get_hash_state_only():
    hash_states = [
        ward.get_hash_state_only(
            state=ward.empty_state,
            patient_type=p
        ) for p in range(3)
    ]
    assert hash_states == [0, 10000, 20000]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p
        ) for p in range(3)
    ]
    assert hash_states == [100000, 110000, 120000]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p
        ) for p in range(3)
    ]
    assert hash_states == [900000, 910000, 920000]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p
        ) for p in range(3)
    ]
    assert hash_states == [29400000, 29410000, 29420000]

    S = np.array(
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_state = ward.get_hash_state_only(
        state=S,
        patient_type=2
    )
    assert hash_state == 510027366320000

def test_get_representative_hash_state():
    # First define some states all in the same equivalence class.
    S = np.array(
        (1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    ST1 = np.array(
        (0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    ST2 = np.array(
        (1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    ST5 = np.array(
        (0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2)
    )
    ST3 = np.array(
        (1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    ST4T6 = np.array(
        (1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    ST5T6 = np.array(
        (0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2)
    )
    ST1T5T6 = np.array(
        (0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2)
    )
    buffer_state = np.zeros(45, dtype=np.int64)
    hS1, idx1 = ward.get_representative_hash_state(state=S, patient_type=1, buffer_state=buffer_state)
    hS2, idx2 = ward.get_representative_hash_state(state=ST1, patient_type=1, buffer_state=buffer_state)
    hS3, idx3 = ward.get_representative_hash_state(state=ST2, patient_type=1, buffer_state=buffer_state)
    hS4, idx4 = ward.get_representative_hash_state(state=ST5, patient_type=1, buffer_state=buffer_state)
    hS5, idx5 = ward.get_representative_hash_state(state=ST3, patient_type=1, buffer_state=buffer_state)
    hS6, idx6 = ward.get_representative_hash_state(state=ST4T6, patient_type=1, buffer_state=buffer_state)
    hS7, idx7 = ward.get_representative_hash_state(state=ST5T6, patient_type=1, buffer_state=buffer_state)
    hS8, idx8 = ward.get_representative_hash_state(state=ST1T5T6, patient_type=1, buffer_state=buffer_state)
    assert hS1 == hS2
    assert hS1 == hS3
    assert hS1 == hS4
    assert hS1 == hS5
    assert hS1 == hS6
    assert hS1 == hS7
    assert hS1 == hS8

    # Now some states not in the same equivalence class.
    Z1 = np.array(
        (1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    Z2 = np.array(
        (1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    Z3 = np.array(
        (1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
    )
    hZ1, idx1 = ward.get_representative_hash_state(state=Z1, patient_type=1, buffer_state=buffer_state)
    hZ2, idx2 = ward.get_representative_hash_state(state=Z2, patient_type=1, buffer_state=buffer_state)
    hZ3, idx3 = ward.get_representative_hash_state(state=Z3, patient_type=1, buffer_state=buffer_state)

    assert hZ1 != hS1
    assert hZ2 != hS1
    assert hZ3 != hS1


def test_dehash_state():
    hash_states = [
        ward.get_hash_state_only(
            state=ward.empty_state,
            patient_type=p
        ) for p in range(3)
    ]
    assert hash_states == [0, 10000, 20000]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p
        ) for p in range(3)
    ]
    assert hash_states == [100000, 110000, 120000]
    for i in range(3):
        state, p = ward.dehash_state(hash_states[i])
        assert p == i
        assert np.array_equal(state, S)

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p
        ) for p in range(3)
    ]
    assert hash_states == [900000, 910000, 920000]
    for i in range(3):
        state, p = ward.dehash_state(hash_states[i])
        assert p == i
        assert np.array_equal(state, S)

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p
        ) for p in range(3)
    ]
    assert hash_states == [29400000, 29410000, 29420000]
    for i in range(3):
        state, p = ward.dehash_state(hash_states[i])
        assert p == i
        assert np.array_equal(state, S)

    S = np.array(
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_state = ward.get_hash_state_only(
        state=S,
        patient_type=2
    )
    assert hash_state == 510027366320000
    state, p = ward.dehash_state(510027366320000)
    assert p == 2
    assert np.array_equal(state, S)


def test_get_hash_stateaction():
    buffer_state = np.zeros(45, dtype=np.int64)
    S = np.array(
        (0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0,
         0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    Shash, idx = ward.get_hash_stateaction(
        state=S,
        patient_type=2,
        action=1102,
        buffer_state=buffer_state
    )
    assert Shash == 103094401121102
    assert idx == 0

    hash_states = [
        ward.get_hash_stateaction(
            state=ward.empty_state,
            patient_type=1,
            action=(a * 101),
            buffer_state=buffer_state
        )[0] for a in range(9)
    ]
    assert hash_states == [10000, 10101, 10202, 10303, 10404, 10505, 10606, 10707, 10808]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    hash_states = [
        ward.get_hash_stateaction(
            state=S,
            patient_type=2,
            action=np.array(a),
            buffer_state=buffer_state
        )[0] for a in [101, 102, 103, 201, 202, 203, 1101, 1102, 1103]
    ]
    assert hash_states == [120101, 120102, 120103, 120201, 120202, 120203, 121101, 121102, 121103]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    ) # The represenative state for this will have [11, 12, 13] flipped to [13, 12, 11]
    hash_states = [
        ward.get_hash_stateaction(
            state=S,
            patient_type=0,
            action=np.array(a),
            buffer_state=buffer_state
        )[0] for a in [101, 202, 1313, 1101, 1112, 1113] # So these actions will be flipped too
    ]
    assert hash_states == [1900101, 1900202, 1901111, 1901301, 1901312, 1901311]


def test_get_state_action_from_hashstate():
    s, a = ward.get_state_action_from_hashstate(9996663331221)
    assert s == 9996663330000
    assert a == 1221
    s, a = ward.get_state_action_from_hashstate(8884442225051)
    assert s == 8884442220000
    assert a == 5051
    s, a = ward.get_state_action_from_hashstate(7773331110810)
    assert s == 7773331110000
    assert a == 810
    s, a = ward.get_state_action_from_hashstate(8883338883331140)
    assert s == 8883338883330000
    assert a == 1140
    s, a = ward.get_state_action_from_hashstate(123456789123565)
    assert s == 123456789120000
    assert a == 3565


def test_inverse_action():
    # First consider transform T_1 and T_3 together. That is (0, 0, 0, 1, 0, 1) = 2^2 + 2^0 = 5
    a = ward.inverse_action(a=0, equivalence_idx=5)
    assert a == 303
    a = ward.inverse_action(a=100, equivalence_idx=5)
    assert a == 203
    a = ward.inverse_action(a=200, equivalence_idx=5)
    assert a == 103
    a = ward.inverse_action(a=300, equivalence_idx=5)
    assert a == 3
    a = ward.inverse_action(a=205, equivalence_idx=5)
    assert a == 105
    a = ward.inverse_action(a=306, equivalence_idx=5)
    assert a == 6
    a = ward.inverse_action(a=8, equivalence_idx=5)
    assert a == 309
    a = ward.inverse_action(a=509, equivalence_idx=5)
    assert a == 508
    a = ward.inverse_action(a=316, equivalence_idx=5)
    assert a == 16
    a = ward.inverse_action(a=616, equivalence_idx=5)
    assert a == 616
    # Now consider transform T_5, T_4 and T_2 together. That is (0, 1, 1, 0, 1, 0) = 2^4 + 2^3 + 2^1 = 26
    a = ward.inverse_action(a=2, equivalence_idx=26)
    assert a == 709
    a = ward.inverse_action(a=615, equivalence_idx=26)
    assert a == 415
    a = ward.inverse_action(a=310, equivalence_idx=26)
    assert a == 1003
    a = ward.inverse_action(a=1406, equivalence_idx=26)
    assert a == 1404
    a = ward.inverse_action(a=1302, equivalence_idx=26)
    assert a == 1109
    # Now consider transform T_5 and T_4 together: That is (0, 1, 1, 0, 0, 0) = 2^4 + 2^3 = 24
    a = ward.inverse_action(a=2, equivalence_idx=24)
    assert a == 709
    a = ward.inverse_action(a=1113, equivalence_idx=24)
    assert a == 1311
    a = ward.inverse_action(a=1112, equivalence_idx=24)
    assert a == 1312
    a = ward.inverse_action(a=400, equivalence_idx=24)
    assert a == 407
    a = ward.inverse_action(a=107, equivalence_idx=24)
    assert a == 800
    a = ward.inverse_action(a=915, equivalence_idx=24)
    assert a == 215
    a = ward.inverse_action(a=1106, equivalence_idx=24)
    assert a == 1306
    a = ward.inverse_action(a=1008, equivalence_idx=24)
    assert a == 301


def test_get_resource_use_per_time_unit():
    S = np.array(
        (1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    assert ward.get_resource_use_per_time_unit(S) == 10

    S = np.array(
        (1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    assert ward.get_resource_use_per_time_unit(S) == 9

    S = np.array(
        (1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    assert ward.get_resource_use_per_time_unit(S) == 10

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2)
    )
    assert ward.get_resource_use_per_time_unit(S) == 16

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 0

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 1

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 1

    S = np.array(
        (0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 2

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 2


def test_get_penalty_per_time_unit():
    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 100

    S = np.array(
        (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 300

    S = np.array(
        (0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 200

    S = np.array(
        (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 1400


def test_get_move_penalty():
    move_penalties = np.array(
        [
            [5.0, 6.0, 7.0],
            [5.5, 6.5, 7.5],
            [0.0, 0.0, 0.0]
        ]
    )
    assert ward.get_move_penalty(0, 1, 0, 0, move_penalties, 33.0) == 5.0
    assert ward.get_move_penalty(0, 5, 0, 0, move_penalties, 33.0) == 5.5
    assert ward.get_move_penalty(0, 14, 0, 0, move_penalties, 33.0) == 5.5
    assert ward.get_move_penalty(0, 1, 1, 0, move_penalties, 33.0) == 6.0
    assert ward.get_move_penalty(0, 5, 1, 0, move_penalties, 33.0) == 6.5
    assert ward.get_move_penalty(0, 14, 1, 0, move_penalties, 33.0) == 6.5
    assert ward.get_move_penalty(0, 1, 2, 0, move_penalties, 33.0) == 7.0
    assert ward.get_move_penalty(0, 5, 2, 0, move_penalties, 33.0) == 7.5
    assert ward.get_move_penalty(0, 14, 2, 0, move_penalties, 33.0) == 7.5
    assert ward.get_move_penalty(7, 7, 1, 1, move_penalties, 33.0) == 0.0
    assert ward.get_move_penalty(1, 0, 0, 0, move_penalties, 33.0) == 5.0
    assert ward.get_move_penalty(5, 0, 0, 0, move_penalties, 33.0) == 5.5
    assert ward.get_move_penalty(14, 0, 0, 0, move_penalties, 33.0) == 5.5
    assert ward.get_move_penalty(1, 0, 1, 0, move_penalties, 33.0) == 6.0
    assert ward.get_move_penalty(5, 0, 1, 0, move_penalties, 33.0) == 6.5
    assert ward.get_move_penalty(1, 0, 2, 0, move_penalties, 33.0) == 7.0
    assert ward.get_move_penalty(5, 0, 2, 0, move_penalties, 33.0) == 7.5
    assert ward.get_move_penalty(0, 15, 0, 1, move_penalties, 33.0) == 33.0
    assert ward.get_move_penalty(5, 15, 0, 1, move_penalties, 33.0) == 33.0
    assert ward.get_move_penalty(14, 15, 0, 1, move_penalties, 33.0) == 33.0


def test_insert_patient():
    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.insert_patient(S, 0, 0)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1)
    )
    ward.insert_patient(S, 2, 14)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.insert_patient(S, 1, 14)
    assert np.array_equal(S, expected_newS)


def test_remove_patient():
    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.remove_patient(S, 0, 1)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.remove_patient(S, 2, 7)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.remove_patient(S, 1, 14)
    assert np.array_equal(S, expected_newS)


def test_move_patient():
    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.move_patient(S, patient_type=0, to_block=0, from_block=3)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.move_patient(S, patient_type=2, to_block=4, from_block=7)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.move_patient(S, patient_type=1, to_block=0, from_block=14)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.move_patient(S, patient_type=0, to_block=0, from_block=14)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.move_patient(S, patient_type=0, to_block=15, from_block=1)
    assert np.array_equal(S, expected_newS)


def test_deteriorate_patient():
    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.deteriorate_patient(S, 0, 1)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0)
    )
    ward.deteriorate_patient(S, 1, 12)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1)
    )
    ward.deteriorate_patient(S, 1, 14)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.deteriorate_patient(S, 0, 13)
    assert np.array_equal(S, expected_newS)


def test_improve_patient():
    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.improve_patient(S, 2, 6)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.improve_patient(S, 1, 14)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.improve_patient(S, 1, 2)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.improve_patient(S, 2, 14)
    assert np.array_equal(S, expected_newS)



def test_get_available_noniso_insert_moves():
    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_moves = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    available_moves = ward.get_available_noniso_insert_moves(S)
    assert np.array_equal(expected_moves, available_moves)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_moves = [0, 4, 8, 9, 10]
    available_moves = ward.get_available_noniso_insert_moves(S)
    assert np.array_equal(expected_moves, available_moves)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_moves = [0, 4, 8, 9, 10]
    available_moves = ward.get_available_noniso_insert_moves(S)
    assert np.array_equal(expected_moves, available_moves)

def test_dehash_action():
    a1, a2 = ward.dehash_action(1212)
    assert a1 == 12
    assert a2 == 12
    a1, a2 = ward.dehash_action(1005)
    assert a1 == 10
    assert a2 == 5
    a1, a2 = ward.dehash_action(510)
    assert a1 == 5
    assert a2 == 10
    a1, a2 = ward.dehash_action(900)
    assert a1 == 9
    assert a2 == 0
    a1, a2 = ward.dehash_action(7)
    assert a1 == 0
    assert a2 == 7
    a1, a2 = ward.dehash_action(0)
    assert a1 == 0
    assert a2 == 0


def test_get_available_actions():
    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    actions_pool = np.empty(16 * 17, dtype=np.int32)
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([0, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([0, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1414], dtype=np.int32))

    S = np.array(
        (1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    actions_pool = np.empty(16 * 17, dtype=np.int32)
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404, 4, 15, 104, 115, 204, 215, 304, 315, 504, 515, 604, 615, 704, 715, 804, 815, 904, 915, 1004, 1015, 1104, 1115, 1204, 1215, 1304, 1315, 1404, 1415], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1404, 1415], dtype=np.int32))

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404, 4, 104, 204, 304, 504, 604, 704, 804, 904, 1004, 1104, 1204, 1304], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1404], dtype=np.int32))

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404, 4, 104, 204, 304, 504, 604, 704, 804, 904, 1004, 1104, 1204, 1304, 1404], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404, 4, 104, 204, 304, 504, 604, 704, 804, 904, 1004, 1104, 1204, 1304, 1404], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404], dtype=np.int32))


    S = np.array(
        (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
         0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
         0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([15, 315, 615, 915, 1115, 1415], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1415], dtype=np.int32))

    S = np.array(
        (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
         0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1111, 111, 411, 711, 1011, 1211, 211, 511, 811, 1311], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1111, 11, 15, 311, 315, 611, 615, 911, 915, 211, 511, 811, 1311, 1411, 1415], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1411, 1415], dtype=np.int32))

    S = np.array(
        (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 2)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([15, 315, 615, 915], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([15, 315, 615, 915], dtype=np.int32))

    S = np.array(
        (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([0, 101, 202], dtype=np.int32))


def test_find_idx_of_patient_to_move():
    patients_blocks = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 8])
    patients_types =  np.array([0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 1, 1, 0, 1])
    assert 8 == ward.find_idx_of_patient_to_move(block=3, patient_type=2, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 0 == ward.find_idx_of_patient_to_move(block=0, patient_type=0, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 15 == ward.find_idx_of_patient_to_move(block=8, patient_type=1, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 5 == ward.find_idx_of_patient_to_move(block=2, patient_type=0, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 6 == ward.find_idx_of_patient_to_move(block=2, patient_type=1, patients_blocks=patients_blocks, patients_types=patients_types)
