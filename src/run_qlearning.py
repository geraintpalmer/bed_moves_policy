import bedmoves
import ciw
import random
import yaml
import argparse
import numpy as np
import multiprocessing
import os
import tqdm
import time

# Force NumPy/OpenBLAS to use only 1 core per process
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def get_Qs(
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
        QLearning=Q,
        seed=seed
    )
    S.simulate_until_max_time(
        max_time=max_time,
        shared_progress_array=progress_array,
        trial=trial
    )
    return Q.return_Qvals()

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

    epsilon_step = 1.0 / (n_stages - 1)
    epsilons = [(i * epsilon_step) for i in range(n_stages)]
    seed = 0
    
    initial_Qvalues = None
    for stage in range(1, n_stages+1):

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
                epsilons[stage-1],
                initial_Qvalues,
                seeds[t],
                t,
                progress_array
            ) for t in range(trials_per_stage)
        ]

        with multiprocessing.Pool(processes=n_threads) as pool:
            results = [pool.apply_async(get_Qs, args) for args in args_list]
            keys_set = []
            qval_set = []
            hits_set = []
            finished_mask = [False] * trials_per_stage

            with tqdm.tqdm(
                total=(max_time * trials_per_stage),
                desc=f"Stage {stage} (epsilon={round(epsilons[stage-1], 3)})",
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
                            keys_set.append(data[0])
                            qval_set.append(data[1])
                            hits_set.append(data[2])
                            results[i] = None # FREE THE DICTIONARY MEMORY IMMEDIATELY
                            finished_mask[i] = True
                    
                    time.sleep(1) # Don't burn CPU checking the array
                pbar.update((max_time * trials_per_stage) - last_min_progress)

        combined = bedmoves.combine_arrays(keys_set, qval_set, hits_set)
        combined_to_save = np.vstack(combined).T

        filename = f"{args.experiment}/results/stage_{stage}_overall_epsilon_{round(epsilons[stage-1], 3)}.csv"
        np.savetxt(
            filename,
            combined_to_save,
            delimiter=",",
            header="Key,Q,Hits",
            fmt=['%d', '%.32f', '%d']
        )

        initial_Qvalues = np.empty(len(combined[0]), dtype=[('Key', 'i8'), ('Q', 'f8'), ('Hits', 'i4')])
        initial_Qvalues['Key'] = np.array(combined[0])
        initial_Qvalues['Q'] = np.array(combined[1])
        initial_Qvalues['Hits'] = np.array(combined[2])

        seed += trials_per_stage
