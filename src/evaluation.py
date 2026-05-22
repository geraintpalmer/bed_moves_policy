import sim, ward, chooser, rl
import ciw
import yaml
import argparse
import numpy as np
import multiprocessing
import os
import tqdm
import time
import gc
import pandas as pd
import ctypes
from ctypes.util import find_library

def trim_memory():
    libc_path = find_library('c')
    if libc_path:
        libc = ctypes.CDLL(libc_path)
        if hasattr(libc, 'malloc_trim'):
            libc.malloc_trim(0)

# Force NumPy/OpenBLAS to use only 1 core per process
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def evaluate(
    max_time,
    occupancy_arrival_probs,
    epsilon,
    initial_keys_path,
    initial_policy_path,
    warmup,
    seed,
    trial,
    progress_array,
):
    """
    Runs
    """
    if initial_keys_path is not None:
        initial_keys = np.memmap(initial_keys_path, dtype=np.int64, mode='r')
        initial_policy = np.memmap(initial_policy_path, dtype=np.int16, mode='r')
    else:
        initial_keys = None
        initial_policy = None

    S = sim.WardEvaluation(
        arrival_distributions=[
            ciw.dists.Exponential(rate=3.0),
            ciw.dists.Exponential(rate=2.0),
            ciw.dists.Exponential(rate=1.0)
        ],
        los_distributions=[
            ciw.dists.Exponential(rate=0.3),
            ciw.dists.Exponential(rate=0.7),
            ciw.dists.Exponential(rate=0.4)
        ],
        deterioration_distributions=[
            ciw.dists.Deterministic(value=np.inf),
            ciw.dists.Deterministic(value=np.inf)
        ],
        improvement_distributions=[
            ciw.dists.Deterministic(value=np.inf),
            ciw.dists.Deterministic(value=np.inf)
        ],
        occupancy_arrival_probs=occupancy_arrival_probs,
        isolation_penalty=8.0,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=15.0,
        epsilon=epsilon,
        seed=seed,
        max_time=max_time,
        initial_keys=initial_keys,
        initial_policy=initial_policy,
        warmup=warmup
    )
    S.simulate_until_max_time(
        shared_progress_array=progress_array,
        trial=trial
    )
    return S.overall_cost - S.warmup_cost

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='The path to the experiment folder.')
    parser.add_argument('n_threads', help='The number of parallel processers to use.')
    args = parser.parse_args()

    with open(args.experiment + "/params_eval.yml") as f:
        params_raw = f.read()
        params = yaml.safe_load(params_raw)

    n_stages = int(params['n_stages'])
    trials_per_stage = int(params['trials_per_stage'])
    max_time = float(params['max_time'])
    warmup = float(params['warmup'])
    n_threads = int(args.n_threads)

    occupancy_arrival_probs = np.genfromtxt('data/state_dependent_arrivals.csv')

    epsilon_step = 1.0 / (n_stages - 1)
    training_epsilons = [(i * epsilon_step) for i in range(n_stages)]
    seed = 0

    eval_epsilons = [0.0] + [1.0 for _ in range(n_stages)]
    
    costs = {}
    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    
    for stage in range(n_stages+1):
        if stage > 0:
            keys = np.memmap(f"{args.experiment}/results/stage_{stage}_overall_keys_epsilon_{round(training_epsilons[stage-1], 3)}.bin", dtype=np.int64, mode='r')
            qvals = np.memmap(f"{args.experiment}/results/stage_{stage}_overall_qvals_epsilon_{round(training_epsilons[stage-1], 3)}.bin", dtype=np.float32, mode='r')
            policy_keys, policy_actions = rl.initialise_policy(
                keys_array=keys,
                qval_array=qvals
            )
            policy_keys_path = f"{args.experiment}/results/stage_{stage}_overall_policykeys_epsilon_{round(training_epsilons[stage-1], 3)}.bin"
            policy_keys.tofile(policy_keys_path)
            policy_actions_path = f"{args.experiment}/results/stage_{stage}_overall_policyactions_epsilon_{round(training_epsilons[stage-1], 3)}.bin"
            policy_actions.tofile(policy_actions_path)
        else:
            policy_keys_path = None
            policy_actions_path = None

        progress_array = manager.Array('d', [0.0] * trials_per_stage)
        seeds = [seed + trial for trial in range(trials_per_stage)]
        args_list = [
            (
                max_time,
                occupancy_arrival_probs,
                eval_epsilons[stage],
                policy_keys_path,
                policy_actions_path,
                warmup,
                seeds[t],
                t,
                progress_array
            ) for t in range(trials_per_stage)
        ]
        costs[f'Stage {stage}'] = []

        with multiprocessing.Pool(processes=n_threads) as pool:
            results = [pool.apply_async(evaluate, args) for args in args_list]
            del args_list
            gc.collect()
            finished_mask = [False] * trials_per_stage

            with tqdm.tqdm(
                total=(max_time * trials_per_stage),
                desc=f"Evaluating Stage {stage} (epsilon={round(eval_epsilons[stage], 3)})",
                unit_scale=True,
                bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]"
            ) as pbar:
                last_min_progress = 0
                while not all(finished_mask):
                    current_min = sum(progress_array)
                    
                    if current_min > last_min_progress:
                        pbar.update(current_min - last_min_progress)
                        last_min_progress = current_min

                    for i, res in enumerate(results):
                        if not finished_mask[i] and res.ready():
                            costs[f'Stage {stage}'].append(res.get())
                            results[i] = None # FREE THE DICTIONARY MEMORY IMMEDIATELY
                            finished_mask[i] = True
                            gc.collect()
                            trim_memory()

                    time.sleep(1) # Don't burn CPU checking the array
                pbar.update((max_time * trials_per_stage) - last_min_progress)

        seed += trials_per_stage

    df = pd.DataFrame(costs)
    df.to_csv(f"{args.experiment}/results/evaluation.csv", index=False)
