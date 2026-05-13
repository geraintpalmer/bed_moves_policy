import ciw
import matplotlib.pyplot as plt
import numpy as np
import tqdm

def get_state_probabilities(seed, max_time, warmup):
    ciw.seed(seed)
    N = ciw.create_network(
        arrival_distributions={
            'Stage 2': [ciw.dists.Exponential(rate=3.0), None, None],
            'Stage 3': [ciw.dists.Exponential(rate=2.0), None, None],
            'Stage 3-I': [ciw.dists.Exponential(rate=1.0), None, None]
        },
        service_distributions={
            'Stage 2': [ciw.dists.Deterministic(value=0.0), ciw.dists.Exponential(rate=0.3), ciw.dists.Exponential(rate=0.3)],
            'Stage 3': [ciw.dists.Deterministic(value=0.0), ciw.dists.Exponential(rate=0.7), ciw.dists.Exponential(rate=0.7)],
            'Stage 3-I': [ciw.dists.Deterministic(value=0.0), ciw.dists.Exponential(rate=0.4), ciw.dists.Exponential(rate=0.4)]
        },
        number_of_servers=[float('inf'), 17, 17],
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
        }
    )
    
    Q = ciw.Simulation(N, tracker=ciw.trackers.NodePopulationSubset([1, 2]))
    Q.simulate_until_max_time(max_time)
    probs = Q.statetracker.state_probabilities(observation_period=(warmup, max_time - warmup))
    
    prob_less_than = {i: 0.0 for i in range(18)}
    prob_equal = {i: 0.0 for i in range(18)}
    prob_total = {i: 0.0 for i in range(18)}
    for n1 in range(18):
        for n2 in range(18):
            p = probs.get((n1, n2), 0.0)
            prob_total[n1] += p
            if n1 < n2:
                prob_less_than[n1] += p
            if n1 == n2:
                prob_equal[n1] += p
    
    for n in range(18):
        if prob_total[n] != 0:
            prob_less_than[n] /= prob_total[n]
            prob_equal[n] /= prob_total[n]
    
    p = [prob_less_than[n] + (prob_equal[n] / 2) for n in range(18)]
    return np.array(p)


n_trials = 100
max_time = 20000
warmup = 100
p = np.zeros(18)
for trial in tqdm.tqdm(range(n_trials)):
    p = p + get_state_probabilities(trial, max_time, warmup)
p = p / n_trials
current_mean = np.mean(p)
diff = 0.5 - current_mean
p = p + diff

np.savetxt("data/state_dependent_arrivals.csv", p, delimiter=",")

fig, ax = plt.subplots(1)
ax.bar(range(18), p, color='darkorange', edgecolor='black')
ax.set_xticks(range(18))
ax.set_xlabel(r"Occupancy ($x$)")
ax.set_ylabel(r"Proportion of arrivals ($p(x)$)")
fig.savefig("../plt/sqa.pdf")
