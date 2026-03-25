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
import numpy as np

def get_Qs(
    max_time,
    learning_rate,
    discount_factor,
    transform_parameter,
    epsilon,
    initial_Qvalues,
    seed
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
        progress_bar=True,
    )
    Q.merge_qvals()
    return (Q.keys, Q.qvals, Q.hits)

def evaluate(
    max_time,
    learning_rate,
    discount_factor,
    transform_parameter,
    epsilon,
    Qvalues,
    seed,
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
        QLearning=Q,
        seed=seed
    )
    S.simulate_until_max_time(max_time=max_time, progress_bar=True)
    return S.overall_cost

if __name__ == '__main__':
    stage = 11
    max_time = 1000
    learning_rate = 0.5
    discount_factor = 0.95
    transform_parameter = 2.0
    epsilon = 0.5
    seed = 0

    data = np.genfromtxt("experiments/exp2/results/stage_11_overall_epsilon_0.5.csv", delimiter=',', dtype=['i8', 'f8', 'i4'], names=True)
    initial_Qvalues = (data['Key'], data['Q'], data['Hits'])
    
    results = get_Qs(max_time, learning_rate, discount_factor, transform_parameter, epsilon, initial_Qvalues, seed)
    eval = evaluate(max_time, learning_rate, discount_factor, transform_parameter, epsilon, initial_Qvalues, seed)

