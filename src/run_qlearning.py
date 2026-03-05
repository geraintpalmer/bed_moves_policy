import bedmoves
import ciw
import random
import pandas as pd
from multiprocessing.pool import ThreadPool
import threading
from pathlib import Path
import yaml
import argparse
import time

def get_Qs(
    max_time,
    learning_rate,
    discount_factor,
    transform_parameter,
    epsilon,
    initial_Qvalues,
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
        initial_Qvalues=initial_Qvalues
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
    Q.update_Qvals_df()
    return Q.Qvals_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='The path to the experiment folder.')
    args = parser.parse_args()

    with open(args.experiment + "/params.yml") as f:
        params_raw = f.read()
        params = yaml.safe_load(params_raw)

    n_stages = int(params['n_stages'])
    trials_per_stage = int(params['trials_per_stage'])
    max_time = float(params['max_time'])
    learning_rate = float(params['learning_rate'])
    discount_factor = float(params['discount_factor'])
    transform_parameter = float(params['transform_parameter'])
    n_threads = int(params['n_threads'])
    write_trials_data = params['write_trials_data']

    epsilon_step = 1.0 / (n_stages - 1)
    epsilons = [(i * epsilon_step) for i in range(n_stages)]
    seed = 0

    tstart = time.time()
    
    initial_Qvalues = None
    for stage in range(1, n_stages+1):
        print(f"====-Stage {stage} (epsilon={round(epsilons[stage-1], 3)})-====")

        seeds = [seed + trial for trial in range(trials_per_stage)]        
        pool = ThreadPool(n_threads)
        lock = threading.Lock()
        results = [pool.apply_async(get_Qs, args=(max_time, learning_rate, discount_factor, transform_parameter, epsilons[stage-1], initial_Qvalues, seeds[trial], trial, lock, f"[Stage {stage}; trial {trial}]")) for trial in range(trials_per_stage)]
        pool.close()
        pool.join()

        stage_dfs = [res.get() for res in results]
        stage_dfs_concat = pd.concat(stage_dfs, axis=1, keys=[f'Trial {i}' for i in range(trials_per_stage)])
        if write_trials_data:
            stage_dfs_concat.to_csv(f"{args.experiment}/results/stage_{stage}_trials_epsilon_{round(epsilons[stage-1], 3)}.csv", index=True)
        initial_Qvalues = bedmoves.combine_Qvalues(stage_dfs)
        initial_Qvalues.to_csv(f"{args.experiment}/results/stage_{stage}_overall_epsilon_{round(epsilons[stage-1], 3)}.csv", index=True)
        seed += n_stages

    tend = time.time()

    print(tstart - tend)

