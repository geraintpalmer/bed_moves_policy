import sim, ward, chooser, rl
import yaml
import argparse
import numpy as np
import multiprocessing
import os
import tqdm
import time
import gc
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
    occupancy_arrival_probs,
    learning_rate,
    discount_factor,
    selection_policy,
    epsilon,
    initial_keys_path,
    initial_qvals_path,
    seed,
    trial,
    progress_array,
    M,
    experiment
):
    """
    Runs
    """
    if initial_keys_path is not None:
        initial_keys = np.memmap(initial_keys_path, dtype=np.int64, mode='r')
        initial_qvals = np.memmap(initial_qvals_path, dtype=np.float32, mode='r')
    else:
        initial_keys = None
        initial_qvals = None

    S = sim.WardTraining(
        arrival_distributions=[
            ('Exponential', 3.0),
            ('Exponential', 2.0),
            ('Exponential', 1.0)
        ],
        los_distributions=[
            ('Exponential', 0.3),
            ('Exponential', 0.7),
            ('Exponential', 0.4)
        ],
        deterioration_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        improvement_distributions=[
            ('Deterministic', np.inf),
            ('Deterministic', np.inf)
        ],
        occupancy_arrival_probs=occupancy_arrival_probs,
        isolation_penalty=8.0,
        move_penalties=np.array([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5]]),
        surge_penalty=15.0,
        selection_policy=selection_policy,
        epsilon=epsilon,
        seed=seed,
        max_time=max_time,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        initial_keys=initial_keys,
        initial_qvals=initial_qvals,
        M=M
    )
    
    if initial_keys is not None:
        del initial_keys
        del initial_qvals

    S.simulate_until_max_time(
        shared_progress_array=progress_array,
        trial=trial
    )
    max_idx, states_array, qval_array, hits_array = S.return_Qvals()
    del S
    gc.collect()
    states_filename = f"{experiment}/results/tmp/states_trial_{trial}.bin"
    qval_filename = f"{experiment}/results/tmp/qvals_trial_{trial}.bin"
    hits_filename = f"{experiment}/results/tmp/hits_trial_{trial}.bin"
    states_array.tofile(states_filename)
    del states_array
    gc.collect()
    qval_array.tofile(qval_filename)
    del qval_array
    gc.collect()
    hits_array.tofile(hits_filename)
    del hits_array
    gc.collect()
    return max_idx

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
    max_epsilon = float(params['max_epsilon'])
    if params['selection_policy'] == 'epsilon_greedy':
        selection_policy = chooser.EPSILON_GREEDY
    if params['selection_policy'] == 'mixture':
        selection_policy = chooser.MIXTURE

    occupancy_arrival_probs = np.genfromtxt('data/state_dependent_arrivals.csv')

    epsilons = rl.get_param_schedule(n_stages=n_stages, max_value=max_epsilon)
    seed = 0

    states_per_stage = np.zeros(n_stages, dtype=np.int64)
    states_after_pruning_per_stage = np.zeros(n_stages, dtype=np.int64)
    
    keys = np.array([], dtype=np.int64)
    qvals = np.array([], dtype=np.float32)
    hits = np.array([], dtype=np.int16)
    keys_path = None
    qvals_path = None

    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    M = None
    prev_key_length = 0
    
    for stage in range(1, n_stages+1):
        progress_array = manager.Array('d', [0.0] * trials_per_stage)
        seeds = [seed + trial for trial in range(trials_per_stage)]
        args_list = [
            (
                max_time,
                occupancy_arrival_probs,
                learning_rate,
                discount_factor,
                selection_policy,
                epsilons[stage-1],
                keys_path,
                qvals_path,
                seeds[t],
                t,
                progress_array,
                M,
                args.experiment
            ) for t in range(trials_per_stage)
        ]

        with multiprocessing.Pool(processes=n_threads, maxtasksperchild=1) as pool:
            results = [pool.apply_async(train, args) for args in args_list]
            del args_list
            gc.collect(2)
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
                            j = res.get()

                            new_states = np.memmap(f"{args.experiment}results/tmp/states_trial_{i}.bin", dtype=np.int64)
                            new_qvals = np.memmap(f"{args.experiment}results/tmp/qvals_trial_{i}.bin", dtype=np.float32)
                            new_hits = np.memmap(f"{args.experiment}results/tmp/hits_trial_{i}.bin", dtype=np.int16)

                            rl.update_master_head_inplace(
                                qvals[:prev_key_length], hits[:prev_key_length], new_qvals[:prev_key_length], new_hits[:prev_key_length]
                            )
                            new_states_tail = new_states[prev_key_length:].copy()
                            new_qvals_tail  = new_qvals[prev_key_length:].copy()
                            new_hits_tail   = new_hits[prev_key_length:].copy()
                            new_states = None
                            new_qvals = None
                            new_hits = None
                            os.remove(f"{args.experiment}results/tmp/states_trial_{i}.bin")
                            os.remove(f"{args.experiment}results/tmp/qvals_trial_{i}.bin")
                            os.remove(f"{args.experiment}results/tmp/hits_trial_{i}.bin")
                            gc.collect(2)

                            keyst, qvalst, hitst = rl.get_unique_tails(keys[prev_key_length:], qvals[prev_key_length:], hits[prev_key_length:], new_states_tail, new_qvals_tail, new_hits_tail)
                            del new_states_tail
                            del new_qvals_tail
                            del new_hits_tail
                            gc.collect(2)

                            keys_temp = np.empty(prev_key_length + len(keyst), dtype=np.int64)
                            keys_temp[:prev_key_length] = keys[:prev_key_length]
                            keys_temp[prev_key_length:] = keyst
                            keys = keys_temp
                            del keyst
                            gc.collect(2)
                            qvals_temp = np.empty(prev_key_length + len(qvalst), dtype=np.float32)
                            qvals_temp[:prev_key_length] = qvals[:prev_key_length]
                            qvals_temp[prev_key_length:] = qvalst
                            qvals = qvals_temp
                            del qvalst
                            gc.collect(2)
                            hits_temp = np.empty(prev_key_length + len(hitst), dtype=np.int16)
                            hits_temp[:prev_key_length] = hits[:prev_key_length]
                            hits_temp[prev_key_length:] = hitst
                            hits = hits_temp
                            del hitst
                            gc.collect(2)

                            results[i] = None # FREE THE DICTIONARY MEMORY IMMEDIATELY
                            finished_mask[i] = True
                            gc.collect(2)
                            trim_memory()
                    
                    time.sleep(1) # Don't burn CPU checking the array
                pbar.update((max_time * trials_per_stage) - last_min_progress)

        gc.collect(2)

        key_length = len(keys)
        states_per_stage[stage-1] = key_length

        keys_path = f"{args.experiment}/results/stage_{stage}_overall_keys_epsilon_{round(epsilons[stage-1], 3)}.bin"
        qvals_path = f"{args.experiment}/results/stage_{stage}_overall_qvals_epsilon_{round(epsilons[stage-1], 3)}.bin"
        key_length = rl.prune_and_save(
            keys_fname=keys_path,
            qval_fname=qvals_path,
            keys=keys,
            qval=qvals,
            hits=hits,
            prune_limit=-1
        )
        keys = np.fromfile(keys_path, dtype=np.int64)
        qvals = np.fromfile(qvals_path, dtype=np.float32)
        hits.fill(np.int16(0))

        extra_space = max(key_length - prev_key_length, key_length * 0.2)
        M = np.ceil((key_length + extra_space) * 1.2).astype(np.int64)
        prev_key_length = key_length
        states_after_pruning_per_stage[stage-1] = key_length
        seed += trials_per_stage

    np.savetxt(f"{args.experiment}/results/unique_states.csv", states_per_stage, delimiter=',')
    np.savetxt(f"{args.experiment}/results/unique_states_after_pruning.csv", states_after_pruning_per_stage, delimiter=',')
