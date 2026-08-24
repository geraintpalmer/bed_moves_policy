mkdir experiments/main_run/results
mkdir experiments/main_run/results/tmp
python training.py experiments/main_run/ $1
python evaluation.py experiments/main_run/ $1
python analysis.py experiments/main_run/