import pandas as pd
import matplotlib.pyplot as plt
import yaml
plt.style.use("seaborn-v0_8-whitegrid")
import numpy as np
import argparse
import rl
import chooser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='The path to the experiment folder.')
    args = parser.parse_args()

    with open(args.experiment + "/params.yml") as f:
        params_raw = f.read()
        params = yaml.safe_load(params_raw)

    epsilons = np.array([1.0, 0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.875, 0.85, 0.825, 0.8, 0.75, 0.5, 0.25, 0.0])
    n_stages = len(epsilons)
    
    stage_labels = [fr"Stage {stage} ($\epsilon={round(epsilons[stage-1], 3)})$" for stage in range(1, n_stages + 1)]
    ticklabels = stage_labels

    data = pd.read_csv(args.experiment + "/results/robust_evaluation.csv")

    # Plot evaluation
    fig, ax = plt.subplots(1, figsize=(7, 5))
    viols = ax.violinplot(
        [data[f'Stage {i}'] for i in range(n_stages)],
        showextrema=False,
        vert=False
    )
    boxes = ax.boxplot(
        [data[f'Stage {i}'] for i in range(n_stages)],
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
    fig.savefig(args.experiment + '/results/robust_evaluation.pdf')
