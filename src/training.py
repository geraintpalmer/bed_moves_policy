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

def train(
    max_time,
    learning_rate,
    discount_factor,
    epsilon,
    initial_keys,
    initial_qvals,
    seed,
    trial,
    progress_array,
):
    """
    Runs
    """
    S = sim.WardTraining(
        arrival_distributions=[
            ciw.dists.Exponential(rate=1.5),
            ciw.dists.Exponential(rate=1.0),
            ciw.dists.Exponential(rate=0.5)
        ],
        los_distributions=[
            ciw.dists.Exponential(rate=0.3),
            ciw.dists.Exponential(rate=0.7),
            ciw.dists.Exponential(rate=0.4)
        ],
        deterioration_distributions=[
            ciw.dists.Exponential(rate=0.1),
            ciw.dists.Exponential(rate=0.2)
        ],
        isolation_penalty=8,
        epsilon=epsilon,
        seed=seed,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        initial_keys=initial_keys,
        initial_qvals=initial_qvals
    )
    S.simulate_until_max_time(
        max_time=max_time,
        shared_progress_array=progress_array,
        trial=trial
    )
    return S.return_Qvals()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='The path to the experiment folder.')
    parser.add_argument('n_threads', help='The number of parallel processers to use.')
    args = parser.parse_args()

    with open(args.experiment + "/params.yml") as f:
        params_raw = f.read()
        params = yaml.safe_load(params_raw)

    n_stages = int(params['n_stages'])
    trials_per_stage = int(params['trials_per_stage'])
    max_time = float(params['max_time'])
    learning_rate = float(params['learning_rate'])
    discount_factor = float(params['discount_factor'])
    n_threads = int(args.n_threads)

    epsilon_step = 1.0 / (n_stages - 1)
    epsilons = [(i * epsilon_step) for i in range(n_stages)]
    seed = 0

    unique_states_per_trial = {s: {t: None for t in range(trials_per_stage)} for s in range(1, n_stages+1)}
    unique_states_per_stage = {s: None for s in range(1, n_stages+1)}
    
    keys = None
    qvals = None
    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    
    for stage in range(1, n_stages+1):
        progress_array = manager.Array('d', [0.0] * trials_per_stage)
        seeds = [seed + trial for trial in range(trials_per_stage)]
        args_list = [
            (
                max_time,
                learning_rate,
                discount_factor,
                epsilons[stage-1],
                keys,
                qvals,
                seeds[t],
                t,
                progress_array
            ) for t in range(trials_per_stage)
        ]

        with multiprocessing.Pool(processes=n_threads) as pool:
            results = [pool.apply_async(train, args) for args in args_list]
            del args_list
            gc.collect()
            keys = np.array([], dtype=np.int64)
            qvals = np.array([], dtype=np.float32)
            hits = np.array([], dtype=np.float32)
            finished_mask = [False] * trials_per_stage

            with tqdm.tqdm(
                total=(max_time * trials_per_stage),
                desc=f"Training Stage {stage} (epsilon={round(epsilons[stage-1], 3)})",
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
                            data = res.get()
                            unique_states_per_trial[stage][i] = data[0]
                            new_keys, new_qvals, new_hits = rl.merge_sorted_qvals(
                                keys, qvals, hits, data[1], data[2], data[3]
                            )
                            data = None
                            keys = new_keys.copy()
                            del new_keys
                            qvals = new_qvals.copy()
                            del new_qvals
                            hits = new_hits.copy()
                            del new_hits
                            results[i] = None # FREE THE DICTIONARY MEMORY IMMEDIATELY
                            finished_mask[i] = True
                            gc.collect()
                            trim_memory()
                    
                    time.sleep(1) # Don't burn CPU checking the array
                pbar.update((max_time * trials_per_stage) - last_min_progress)

        filename = f"{args.experiment}/results/stage_{stage}_overall_epsilon_{round(epsilons[stage-1], 3)}.npz"
        np.savez(filename, keys=keys, vals=qvals, hits=hits)
        unique_states_per_stage[stage] = len(keys)

        seed += trials_per_stage

    unique_states = pd.DataFrame(
        {
            f'Stage {s}': [unique_states_per_trial[s][t] for t in range(trials_per_stage)] + [unique_states_per_stage[s]] for s in range(1, n_stages+1)
        }, index=[f'Trial {t}' for t in range(trials_per_stage)] + ['Overall']
    )
    unique_states.to_csv(f"{args.experiment}/results/unique_states.csv")
