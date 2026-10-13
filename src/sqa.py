import ciw
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import yaml

with open("icu_params.yml") as f:
    params_raw = f.read()
    params = yaml.safe_load(params_raw)

lambda_2 = float(params['lambda_2'])
lambda_3 = float(params['lambda_3'])
lambda_3I = float(params['lambda_3I'])
r23 = float(params['r23'])
r33I = float(params['r33I'])
r32 = float(params['r32'])
r3I3 = float(params['r3I3'])
mu_2 = float(params['mu_2'])
mu_3 = float(params['mu_3'])
mu_3I = float(params['mu_3I'])
eta = float(params['eta'])

small_model_scaling = 16 / 17

absorbing_matrix = [
    [-(r23 + mu_2), r23, 0, mu_2],
    [r32, -(r32 + r33I + mu_3), r33I, mu_3],
    [0, r3I3, -(r3I3 + mu_3I), mu_3I],
    [0, 0, 0, 0]
]


def get_state_probabilities(seed, max_time, warmup):
    ciw.seed(seed)
    N = ciw.create_network(
        arrival_distributions={
            'Stage 2': [ciw.dists.Exponential(rate=lambda_2 * eta * small_model_scaling), None, None],
            'Stage 3': [ciw.dists.Exponential(rate=lambda_3 * eta * small_model_scaling), None, None],
            'Stage 3-I': [ciw.dists.Exponential(rate=lambda_3I * eta * small_model_scaling), None, None]
        },
        service_distributions={
            'Stage 2': [
                ciw.dists.Deterministic(value=0.0),
                ciw.dists.PhaseType([1, 0, 0, 0], absorbing_matrix),
                ciw.dists.PhaseType([1, 0, 0, 0], absorbing_matrix)
            ],
            'Stage 3': [
                ciw.dists.Deterministic(value=0.0),
                ciw.dists.PhaseType([0, 1, 0, 0], absorbing_matrix),
                ciw.dists.PhaseType([0, 1, 0, 0], absorbing_matrix)
            ],
            'Stage 3-I': [
                ciw.dists.Deterministic(value=0.0),
                ciw.dists.PhaseType([0, 0, 1, 0], absorbing_matrix),
                ciw.dists.PhaseType([0, 0, 1, 0], absorbing_matrix)
            ]
        },
        number_of_servers=[float('inf'), 16, 16],
        routing={
            'Stage 2': ciw.routing.NetworkRouting(
                routers=[
                    ciw.routing.LoadBalancing(destinations=[2, 3]),
                    ciw.routing.Leave(),
                    ciw.routing.Leave()
                ]
            ),
            'Stage 3': ciw.routing.NetworkRouting(
                routers=[
                    ciw.routing.LoadBalancing(destinations=[2, 3]),
                    ciw.routing.Leave(),
                    ciw.routing.Leave()
                ]
            ),
            'Stage 3-I': ciw.routing.NetworkRouting(
                routers=[
                    ciw.routing.LoadBalancing(destinations=[2, 3]),
                    ciw.routing.Leave(),
                    ciw.routing.Leave()
                ]
            ),
        },
        system_capacity=32
    )
    
    Q = ciw.Simulation(N, tracker=ciw.trackers.NodePopulationSubset([1, 2]))
    Q.simulate_until_max_time(max_time)
    probs = Q.statetracker.state_probabilities(observation_period=(warmup, max_time - warmup))
    
    prob_less_than1 = {i: 0.0 for i in range(17)}
    prob_equal1 = {i: 0.0 for i in range(17)}
    prob_total1 = {i: 0.0 for i in range(17)}
    for n1 in range(17):
        for n2 in range(17):
            p = probs.get((n1, n2), 0.0)
            prob_total1[n1] += p
            if n1 < n2:
                prob_less_than1[n1] += p
            if n1 == n2:
                prob_equal1[n1] += p

    prob_less_than2 = {i: 0.0 for i in range(17)}
    prob_equal2 = {i: 0.0 for i in range(17)}
    prob_total2 = {i: 0.0 for i in range(17)}
    for n2 in range(17):
        for n1 in range(17):
            p = probs.get((n1, n2), 0.0)
            prob_total2[n2] += p
            if n2 < n1:
                prob_less_than2[n2] += p
            if n1 == n2:
                prob_equal2[n2] += p
    
    for n in range(17):
        if prob_total1[n] > 0.0001:
            prob_less_than1[n] /= prob_total1[n]
            prob_equal1[n] /= prob_total1[n]
        if prob_total1[n] > 0.0001:
            prob_less_than2[n] /= prob_total2[n]
            prob_equal2[n] /= prob_total2[n]

    p1 = [prob_less_than1[n] + (prob_equal1[n] / 2) for n in range(17)]
    p2 = [prob_less_than2[n] + (prob_equal2[n] / 2) for n in range(17)]
    p = [(prob1 + prob2) / 2 for prob1, prob2 in zip(p1, p2)]
    return np.array(p)


n_trials = 25
max_time = 50000
warmup = 2000
p = np.zeros(17)
for trial in tqdm.tqdm(range(n_trials)):
    state_probs = get_state_probabilities(trial, max_time, warmup)
    p = p + state_probs
p = p / n_trials

# Enforce Monotonicity for seldom visited states
argmax_p = np.argmax(p)
max_p = max(p)
for i in range(argmax_p):
    p[i] = max_p

np.savetxt("data/state_dependent_arrivals.csv", p, delimiter=",")

fig, ax = plt.subplots(1)
ax.bar(range(17), p, color='darkorange', edgecolor='black')
ax.set_xticks(range(17))
ax.set_xlabel(r"Occupancy ($x$)")
ax.set_ylabel(r"Proportion of arrivals ($p(x)$)")
fig.savefig("../plt/sqa.pdf")
