# mkdir experiments/final_run/results
# mkdir experiments/final_run/results/tmp
# python training.py experiments/final_run/ $1
# python evaluation.py experiments/final_run/ $1
python robust_evaluation.py experiments/final_run/ $1 31

mkdir experiments/final_run41/results
mkdir experiments/final_run41/results/tmp
python training.py experiments/final_run41/ $1
python evaluation.py experiments/final_run41/ $1