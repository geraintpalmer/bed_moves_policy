echo "Scheme = Epsilon-Greedy"
mkdir experiments/scheme_experiment/epsilon_greedy/results
mkdir experiments/scheme_experiment/epsilon_greedy/results/tmp
python training.py experiments/scheme_experiment/epsilon_greedy/ $1
python evaluation.py experiments/scheme_experiment/epsilon_greedy/ $1
