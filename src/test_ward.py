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
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p
        ) for p in range(3)
    ]
    assert hash_states == [100000, 110000, 120000]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p
        ) for p in range(3)
    ]
    assert hash_states == [900000, 910000, 920000]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_state_only(
            state=S,
            patient_type=p
        ) for p in range(3)
    ]
    assert hash_states == [29400000, 29410000, 29420000]

    S = np.array(
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_state = ward.get_hash_state_only(
        state=S,
        patient_type=2
    )
    assert hash_state == 2040109465520000

def test_get_representative_hash_state():
    # First define some states all in the same equivalence class.
    S1 = np.array(
        (1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    S2 = np.array(
        (0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    S3 = np.array(
        (1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    S4 = np.array(
        (1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    S5 = np.array(
        (1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    S6 = np.array(
        (1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    S7 = np.array(
        (0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    S8 = np.array(
        (1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    )
    buffer_state = np.zeros(48, dtype=np.int64)
    hS1, idx1 = ward.get_representative_hash_state(state=S1, patient_type=1, buffer_state=buffer_state)
    hS2, idx2 = ward.get_representative_hash_state(state=S2, patient_type=1, buffer_state=buffer_state)
    hS3, idx3 = ward.get_representative_hash_state(state=S3, patient_type=1, buffer_state=buffer_state)
    hS4, idx4 = ward.get_representative_hash_state(state=S4, patient_type=1, buffer_state=buffer_state)
    hS5, idx5 = ward.get_representative_hash_state(state=S5, patient_type=1, buffer_state=buffer_state)
    hS6, idx6 = ward.get_representative_hash_state(state=S6, patient_type=1, buffer_state=buffer_state)
    hS7, idx7 = ward.get_representative_hash_state(state=S7, patient_type=1, buffer_state=buffer_state)
    hS8, idx8 = ward.get_representative_hash_state(state=S8, patient_type=1, buffer_state=buffer_state)
    assert hS1 == hS2 == hS3 == hS4 == hS5 == hS6 == hS7 == hS8

    # Now some states not in the same equivalence class.
    Z1 = np.array(
        (1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    Z2 = np.array(
        (1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    Z3 = np.array(
        (1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
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
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
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
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
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
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
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
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_state = ward.get_hash_state_only(
        state=S,
        patient_type=2
    )
    assert hash_state == 2040109465520000
    state, p = ward.dehash_state(2040109465520000)
    assert p == 2
    assert np.array_equal(state, S)


def test_get_hash_stateaction():
    buffer_state = np.zeros(48, dtype=np.int64)
    S = np.array(
        (1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    Shash, idx = ward.get_hash_stateaction(
        state=S,
        patient_type=2,
        action=1104,
        buffer_state=buffer_state
    )
    assert Shash == 412377609521104
    assert idx == 26

    hash_states = [
        ward.get_hash_stateaction(
            state=ward.empty_state,
            patient_type=1,
            action=(a * 1111),
            buffer_state=buffer_state
        )[0] for a in range(9)
    ]
    assert hash_states == [10000, 11111, 12222, 13333, 14444, 15555, 16666, 17777, 18888]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    hash_states = [
        ward.get_hash_stateaction(
            state=S,
            patient_type=2,
            action=np.array(a),
            buffer_state=buffer_state
        )[0] for a in [2002, 1212, 2222, 3232, 4242, 5252, 6262, 7272]
    ]
    assert hash_states == [122002, 121212, 122222, 123232, 124242, 125252, 126262, 127272]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_stateaction(
            state=S,
            patient_type=0,
            action=np.array(a),
            buffer_state=buffer_state
        )[0] for a in [  20, 1221, 2222, 3223, 4224, 5225, 6226, 7227,
                       8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]
    ]
    assert hash_states == [
        900020, 901221, 902222, 903223, 904224, 905225, 906226, 907227,
        908000, 908001, 908002, 908003, 908004, 908005, 908006, 908007
    ]

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    hash_states = [
        ward.get_hash_stateaction(
            state=S,
            patient_type=1,
            action=np.array(a),
            buffer_state=buffer_state
        )[0] for a in [ 110, 1111, 2212, 3313, 4414, 5515,
                       6600, 6601, 6602, 6603, 6604, 6605,
                       7700, 7701, 7702, 7703, 7704, 7705,
                       8800, 8801, 8802, 8803, 8804, 8805]
    ]
    assert hash_states == [
        29410110, 29411111, 29412212, 29413313, 29414414, 29415515,
        29416600, 29416601, 29416602, 29416603, 29416604, 29416605,
        29417700, 29417701, 29417702, 29417703, 29417704, 29417705,
        29418800, 29418801, 29418802, 29418803, 29418804, 29418805
    ]


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
    # First consider transform T_1 and T_3 together. That is (0, 0, 1, 0, 1) = 2^2 + 2^0 = 5
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
    a = ward.inverse_action(a=307, equivalence_idx=5)
    assert a == 7
    a = ward.inverse_action(a=8, equivalence_idx=5)
    assert a == 311
    a = ward.inverse_action(a=509, equivalence_idx=5)
    assert a == 510
    a = ward.inverse_action(a=316, equivalence_idx=5)
    assert a == 16
    a = ward.inverse_action(a=616, equivalence_idx=5)
    assert a == 616
    # Now consider transform T_5, T_4 and T_2 together. That is (1, 1, 0, 1, 0) = 2^4 + 2^3 + 2^1 = 26
    a = ward.inverse_action(a=2, equivalence_idx=26)
    assert a == 705
    a = ward.inverse_action(a=716, equivalence_idx=26)
    assert a == 316
    a = ward.inverse_action(a=310, equivalence_idx=26)
    assert a == 410
    a = ward.inverse_action(a=1406, equivalence_idx=26)
    assert a == 1202
    a = ward.inverse_action(a=1502, equivalence_idx=26)
    assert a == 1505
    # Now consider transform T_5 and T_4 together: That is (1, 1, 0, 0, 0) = 2^4 + 2^3 = 24
    a = ward.inverse_action(a=2, equivalence_idx=24)
    assert a == 406
    a = ward.inverse_action(a=1214, equivalence_idx=24)
    assert a == 1412
    a = ward.inverse_action(a=1213, equivalence_idx=24)
    assert a == 1413
    a = ward.inverse_action(a=400, equivalence_idx=24)
    assert a == 4
    a = ward.inverse_action(a=107, equivalence_idx=24)
    assert a == 503
    a = ward.inverse_action(a=915, equivalence_idx=24)
    assert a == 915
    a = ward.inverse_action(a=1206, equivalence_idx=24)
    assert a == 1402
    a = ward.inverse_action(a=1008, equivalence_idx=24)
    assert a == 1008


def test_get_resource_use_per_time_unit():
    S = np.array(
        (1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    assert ward.get_resource_use_per_time_unit(S) == 10

    S = np.array(
        (1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    assert ward.get_resource_use_per_time_unit(S) == 9

    S = np.array(
        (1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2)
    )
    assert ward.get_resource_use_per_time_unit(S) == 10

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2)
    )
    assert ward.get_resource_use_per_time_unit(S) == 17

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 0

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 1

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 1

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 2

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_resource_use_per_time_unit(S) == 2


def test_get_penalty_per_time_unit():
    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 100

    S = np.array(
        (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 0

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 300

    S = np.array(
        (0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 200

    S = np.array(
        (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    )
    assert ward.get_penalty_per_time_unit(S, 100) == 1500


def test_get_move_penalty():
    move_penalties = np.array(
        [
            [5.0, 6.0, 7.0],
            [5.5, 6.5, 7.5],
            [0.0, 0.0, 0.0]
        ]
    )
    assert ward.get_move_penalty(0, 1, 0, 0, move_penalties, 33.0) == 5.0
    assert ward.get_move_penalty(0, 10, 0, 0, move_penalties, 33.0) == 5.5
    assert ward.get_move_penalty(0, 15, 0, 0, move_penalties, 33.0) == 5.5
    assert ward.get_move_penalty(0, 1, 1, 0, move_penalties, 33.0) == 6.0
    assert ward.get_move_penalty(0, 10, 1, 0, move_penalties, 33.0) == 6.5
    assert ward.get_move_penalty(0, 15, 1, 0, move_penalties, 33.0) == 6.5
    assert ward.get_move_penalty(0, 1, 2, 0, move_penalties, 33.0) == 7.0
    assert ward.get_move_penalty(0, 10, 2, 0, move_penalties, 33.0) == 7.5
    assert ward.get_move_penalty(0, 15, 2, 0, move_penalties, 33.0) == 7.5
    assert ward.get_move_penalty(5, 5, 1, 1, move_penalties, 33.0) == 0.0
    assert ward.get_move_penalty(1, 0, 0, 0, move_penalties, 33.0) == 5.0
    assert ward.get_move_penalty(10, 0, 0, 0, move_penalties, 33.0) == 5.5
    assert ward.get_move_penalty(15, 0, 0, 0, move_penalties, 33.0) == 5.5
    assert ward.get_move_penalty(1, 0, 1, 0, move_penalties, 33.0) == 6.0
    assert ward.get_move_penalty(10, 0, 1, 0, move_penalties, 33.0) == 6.5
    assert ward.get_move_penalty(1, 0, 2, 0, move_penalties, 33.0) == 7.0
    assert ward.get_move_penalty(10, 0, 2, 0, move_penalties, 33.0) == 7.5
    assert ward.get_move_penalty(0, 16, 0, 1, move_penalties, 33.0) == 33.0
    assert ward.get_move_penalty(10, 16, 0, 1, move_penalties, 33.0) == 33.0
    assert ward.get_move_penalty(15, 16, 0, 1, move_penalties, 33.0) == 33.0


def test_insert_patient():
    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.insert_patient(S, 0, 0)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    ward.insert_patient(S, 2, 15)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.insert_patient(S, 1, 15)
    assert np.array_equal(S, expected_newS)


def test_remove_patient():
    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.remove_patient(S, 0, 1)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.remove_patient(S, 2, 7)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.remove_patient(S, 1, 15)
    assert np.array_equal(S, expected_newS)


def test_move_patient():
    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.move_patient(S, patient_type=0, to_block=0, from_block=3)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.move_patient(S, patient_type=2, to_block=4, from_block=7)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.move_patient(S, patient_type=1, to_block=0, from_block=15)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.move_patient(S, patient_type=0, to_block=0, from_block=15)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.move_patient(S, patient_type=0, to_block=16, from_block=1)
    assert np.array_equal(S, expected_newS)


def test_deteriorate_patient():
    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.deteriorate_patient(S, 0, 1)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0)
    )
    ward.deteriorate_patient(S, 1, 13)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    ward.deteriorate_patient(S, 1, 15)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.deteriorate_patient(S, 0, 14)
    assert np.array_equal(S, expected_newS)


def test_improve_patient():
    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.improve_patient(S, 2, 6)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.improve_patient(S, 1, 15)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_newS = np.array(
        (0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.improve_patient(S, 1, 2)
    assert np.array_equal(S, expected_newS)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    expected_newS = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    ward.improve_patient(S, 2, 15)
    assert np.array_equal(S, expected_newS)



def test_get_available_noniso_insert_moves():
    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_moves = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    available_moves = ward.get_available_noniso_insert_moves(S)
    assert np.array_equal(expected_moves, available_moves)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_moves = [0, 4, 8, 9, 10, 11]
    available_moves = ward.get_available_noniso_insert_moves(S)
    assert np.array_equal(expected_moves, available_moves)

    S = np.array(
        (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    expected_moves = [0, 4, 8, 9, 10, 11]
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
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    actions_pool = np.empty(16 * 17, dtype=np.int32)
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([0, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([0, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1515], dtype=np.int32))

    S = np.array(
        (1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    actions_pool = np.empty(16 * 17, dtype=np.int32)
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404, 4, 16, 104, 116, 204, 216, 304, 316, 504, 516, 604, 616, 704, 716, 804, 816, 904, 916, 1004, 1016, 1104, 1116, 1204, 1216, 1304, 1316, 1404, 1416, 1504, 1516], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1504, 1516], dtype=np.int32))

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404, 4, 104, 204, 304, 504, 604, 704, 804, 904, 1004, 1104, 1204, 1304, 1404], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1504], dtype=np.int32))

    S = np.array(
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404, 4, 104, 204, 304, 504, 604, 704, 804, 904, 1004, 1104, 1204, 1304, 1404], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404, 4, 104, 204, 304, 504, 604, 704, 804, 904, 1004, 1104, 1204, 1304, 1404], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([404], dtype=np.int32))


    S = np.array(
        (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
         0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
         0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([16, 316, 616, 916, 1216, 1516], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1516], dtype=np.int32))

    S = np.array(
        (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
         0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1212, 112, 412, 712, 1012, 1312, 212, 512, 812, 1112, 1412], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1212, 12, 16, 312, 316, 612, 616, 912, 916, 212, 512, 812, 1112, 1412, 1512, 1516], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([1512, 1516], dtype=np.int32))

    S = np.array(
        (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0,
         0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 2)
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=1, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([16, 316, 616, 916], dtype=np.int32))
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=2, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([16, 316, 616, 916], dtype=np.int32))

    S = np.array(
        (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.int64
    )
    available_moves, valid_count = ward.get_available_actions(state=S, patient_type=0, actions_pool=actions_pool)
    assert np.array_equal(available_moves[:valid_count], np.array([0, 101, 202], dtype=np.int32))


def test_find_idx_of_patient_to_move():
    patients_blocks = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 8, 8])
    patients_types =  np.array([0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 1, 1, 0, 1, 2])
    assert 8 == ward.find_idx_of_patient_to_move(block=3, patient_type=2, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 0 == ward.find_idx_of_patient_to_move(block=0, patient_type=0, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 16 == ward.find_idx_of_patient_to_move(block=8, patient_type=2, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 5 == ward.find_idx_of_patient_to_move(block=2, patient_type=0, patients_blocks=patients_blocks, patients_types=patients_types)
    assert 6 == ward.find_idx_of_patient_to_move(block=2, patient_type=1, patients_blocks=patients_blocks, patients_types=patients_types)
