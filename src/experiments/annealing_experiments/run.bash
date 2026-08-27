mkdir experiments/annealing_experiments/concave/results
mkdir experiments/annealing_experiments/concave/results/tmp
python training.py experiments/annealing_experiments/concave/ $1
python evaluation.py experiments/annealing_experiments/concave/ $1

mkdir experiments/annealing_experiments/convex/results
mkdir experiments/annealing_experiments/convex/results/tmp
python training.py experiments/annealing_experiments/convex/ $1
python evaluation.py experiments/annealing_experiments/convex/ $1
