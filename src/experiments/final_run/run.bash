mkdir experiments/final_run/results
mkdir experiments/final_run/results/tmp
python training.py experiments/final_run/ $1
python evaluation.py experiments/final_run/ $1