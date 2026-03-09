import bedmoves
import ciw
import random
import pandas as pd
from multiprocessing.pool import ThreadPool
import threading
from pathlib import Path
import yaml
import argparse

def evaluate(
    max_time,
    learning_rate,
    discount_factor,
    transform_parameter,
    epsilon,
    Qvalues,
    seed,
    trial,
    lock,
    progress_bar_description
):
    """
    Runs
    """
    Q = bedmoves.QLearning(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        transform_parameter=transform_parameter,
        initial_Qvalues=Qvalues,
        learn=False
    )
    S = bedmoves.BedMoveSimulation(
        arrival_distributions=[
            ciw.dists.Exponential(1.5),
            ciw.dists.Exponential(1.0),
            ciw.dists.Exponential(0.5)
        ],
        los_distributions=[
            ciw.dists.Exponential(0.3),
            ciw.dists.Exponential(0.7),
            ciw.dists.Exponential(0.4)
        ],
        action_chooser=bedmoves.EpsilonHard(
            epsilon=epsilon,
            QLearning=Q
        ),
        isolation_penalty=8,
        adjacent_move_penalty=1,
        nonadjacent_move_penalty=2,
        Qlearning=Q,
        seed=seed
    )
    S.simulate_until_max_time(max_time=max_time, lock=lock, progress_bar=True, progress_bar_description=progress_bar_description)
    return S.overall_cost

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='The path to the experiment folder.')
    args = parser.parse_args()

    with open(args.experiment + "/params_eval.yml") as f:
        params_raw = f.read()
        params = yaml.safe_load(params_raw)

    n_stages = int(params['n_stages'])
    trials_per_stage = int(params['trials_per_stage'])
    max_time = float(params['max_time'])
    learning_rate = float(params['learning_rate'])
    discount_factor = float(params['discount_factor'])
    transform_parameter = float(params['transform_parameter'])
    n_threads = int(params['n_threads'])

    epsilon_step = 1.0 / (n_stages - 1)
    training_epsilons = [(i * epsilon_step) for i in range(n_stages)]
    seed = 0

    eval_epsilons = [0.0] + [1.0 for _ in range(n_stages)]

    costs = {}
    for stage in range(n_stages + 1):
        print(f"====-Stage {stage} (epsilon={round(training_epsilons[stage-1], 3)})-====")
        if stage > 0:
            Qvalues = pd.read_csv(f"{args.experiment}/results/stage_{stage}_overall_epsilon_{round(training_epsilons[stage-1], 3)}.csv", index_col=0)
        else:
            Qvalues = None

        seeds = [seed + trial for trial in range(trials_per_stage)]
        
        pool = ThreadPool(n_threads)
        lock = threading.Lock()
        results = [pool.apply_async(evaluate, args=(max_time, learning_rate, discount_factor, transform_parameter, eval_epsilons[stage], Qvalues, seeds[trial], trial, lock, f"[Stage {stage}; trial {trial}]")) for trial in range(trials_per_stage)]
        pool.close()
        pool.join()

        costs[f'Stage {stage}'] = [res.get() for res in results]
        seed += n_stages

    df = pd.DataFrame(costs)
    df.to_csv(f"{args.experiment}/results/evaluation.csv", index=False)

