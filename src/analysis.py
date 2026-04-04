import pandas as pd
import matplotlib.pyplot as plt
import yaml
plt.style.use("seaborn-v0_8-whitegrid")
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='The path to the experiment folder.')
    args = parser.parse_args()

    with open(args.experiment + "/params_eval.yml") as f:
        params_raw = f.read()
        params = yaml.safe_load(params_raw)

    epsilon_step = 1.0 / (params['n_stages'] - 1)
    epsilons = [(i * epsilon_step) for i in range(params['n_stages'])]
    stage_labels = [f"Stage {stage} (epsilon={round(epsilons[stage-1], 3)})" for stage in range(1, params['n_stages'] + 1)]
    ticklabels = ['Random'] + stage_labels

    data = pd.read_csv(args.experiment + "/results/evaluation.csv")

    unique_states = pd.read_csv(args.experiment + "/results/unique_states.csv", index_col=0)
    visited_states = np.array([unique_states.loc['Overall', f'Stage {s}'] for s in range(1, params['n_stages']+1)])

    # Plot evaluation
    fig, ax = plt.subplots(1, figsize=(7, 5))
    viols = ax.violinplot(
        [data[f'Stage {i}'] for i in range(params['n_stages'] + 1)],
        showextrema=False,
        vert=False
    )
    boxes = ax.boxplot(
        [data[f'Stage {i}'] for i in range(params['n_stages'] + 1)],
        whis=(0, 100),
        showmeans=True,
        medianprops={'color': 'black'},
        meanprops={
            'markerfacecolor':'darkorange',
            'markeredgecolor': 'darkorange',
            'marker': '*'
        },
        vert=False
    )
    
    for pc in viols['bodies']:
        pc.set_facecolor('darkorange')
        pc.set_edgecolor('darkorange')
    
    plt.gca().invert_yaxis()
    
    ax.set_yticklabels(ticklabels)
    ax.set_xlabel("Overall Cost")
    plt.tight_layout()
    fig.savefig(args.experiment + '/results/cost_by_stage.pdf')


    # Progression
    fig, axarr = plt.subplots(1, 2, figsize=(12, 3.5))
    axarr[0].barh(stage_labels, visited_states, color='darkorange', edgecolor='black')
    axarr[0].set_xlabel("Total Visited States")
    plt.gca().invert_yaxis()
    axarr[1].barh(stage_labels, [visited_states[0]] + list(np.diff(visited_states)), color='darkorange', edgecolor='black')
    axarr[1].set_xlabel("New States Visited per Stage")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig.savefig(args.experiment + '/results/unique_visits_per_stage.pdf')
