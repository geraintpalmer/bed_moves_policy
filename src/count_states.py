def number_states(n):
    """
    The number of states when there are n independent non-isolation columns
    """
    return 3 * 10 * (4 ** n)

def number_full_states(n):
    """
    The number of full states when there are n independent non-isolation columns
    """
    return 6 * (3 ** n)

def number_full_states_nostage2(n):
    """
    The number of full states with no stage 2 patients when there are n
    independent non-isolation columns
    """
    return 2 * 3 * (2 ** n)

def total_states(n):
    """
    The overall total number of states (exlcuding states where patients cannot
    arrive) when there are n independent non-isolation columns
    """
    return number_states(n) - number_full_states(n) - number_full_states_nostage2(n)



def burnside_lemma(ns, function):
    """
    Performs bernside's lemma on a given counting function, for a set of ns.
    """
    return sum(function(n) for n in ns) / len(ns)

size_state_space = total_states(14)

fixed_point_ns = [
    14, 12, 13, 12, 13, 10, 11, 11, 10, 11, 8, 9, 11, 12, 9, 10, 11, 8, 9, 9,
    10, 7, 9, 10, 7, 8, 9, 10, 7, 7, 8, 5, 10, 7, 8, 8, 11, 6, 7, 8, 5, 6, 8, 9,
    6, 6, 9, 4, 9, 6, 7, 4, 6, 9, 4, 7, 4, 8, 7, 6, 5, 6, 5, 7
]
size_representative_state_space = burnside_lemma(fixed_point_ns, total_states)

print(f"Size of state space: {size_state_space}\nSize of representative state space: {size_representative_state_space}")
