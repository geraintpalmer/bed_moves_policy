echo "From 0.0 to 0.5"
mkdir experiments/robustness_experiment/0_to_05/results
mkdir experiments/robustness_experiment/0_to_05/results/tmp
python training.py experiments/robustness_experiment/0_to_05/ $1
python evaluation.py experiments/robustness_experiment/0_to_05/ $1

echo "From 0.0 to 0.2"
mkdir experiments/robustness_experiment/0_to_02/results
mkdir experiments/robustness_experiment/0_to_02/results/tmp
python training.py experiments/robustness_experiment/0_to_02/ $1
python evaluation.py experiments/robustness_experiment/0_to_02/ $1
