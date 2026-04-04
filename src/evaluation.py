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

# Force NumPy/OpenBLAS to use only 1 core per process
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def evaluate(
    max_time,
    learning_rate,
    discount_factor,
    transform_parameter,
    epsilon,
    initial_Qvalues,
    seed,
    trial,
    progress_array,
):
    """
    Runs
    """
    S = sim.WardRLSimulation(
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
        isolation_penalty=8,
        epsilon=epsilon,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        transform_parameter=transform_parameter,
        seed=seed,
        initial_Qvalues=initial_Qvalues,
        learn=False
    )
    S.simulate_until_max_time(
        max_time=max_time,
        shared_progress_array=progress_array,
        trial=trial
    )
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
    for stage in range(n_stages+1):
        if stage > 0:
            data = np.load(f"{args.experiment}/results/stage_{stage}_overall_epsilon_{round(training_epsilons[stage-1], 3)}.npz")
            Qvalues =  data['keys'].astype(np.int64), data['vals'].astype(np.float64)
        else:
            Qvalues = None


        multiprocessing.set_start_method("spawn", force=True)
        manager = multiprocessing.Manager()
        progress_array = manager.Array('d', [0.0] * trials_per_stage)

        seeds = [seed + trial for trial in range(trials_per_stage)]
        args_list = [
            (
                max_time,
                learning_rate,
                discount_factor,
                transform_parameter,
                eval_epsilons[stage],
                Qvalues,
                seeds[t],
                t,
                progress_array
            ) for t in range(trials_per_stage)
        ]
        costs[f'Stage {stage}'] = []

        with multiprocessing.Pool(processes=n_threads) as pool:
            results = [pool.apply_async(evaluate, args) for args in args_list]
            finished_mask = [False] * trials_per_stage

            with tqdm.tqdm(
                total=max_time * trials_per_stage,
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

                    time.sleep(1) # Don't burn CPU checking the array
                pbar.update((max_time * trials_per_stage) - last_min_progress)

        seed += trials_per_stage

    df = pd.DataFrame(costs)
    df.to_csv(f"{args.experiment}/results/evaluation.csv", index=False)
