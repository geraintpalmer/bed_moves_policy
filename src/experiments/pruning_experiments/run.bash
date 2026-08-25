echo "h = 1"
mkdir experiments/pruning_experiments/h001/results
mkdir experiments/pruning_experiments/h001/results/tmp
python training.py experiments/pruning_experiments/h001/ $1
python evaluation.py experiments/pruning_experiments/h001/ $1

echo "h = 5"
mkdir experiments/pruning_experiments/h005/results
mkdir experiments/pruning_experiments/h005/results/tmp
python training.py experiments/pruning_experiments/h005/ $1
python evaluation.py experiments/pruning_experiments/h005/ $1

echo "h = 20"
mkdir experiments/pruning_experiments/h020/results
mkdir experiments/pruning_experiments/h020/results/tmp
python training.py experiments/pruning_experiments/h020/ $1
python evaluation.py experiments/pruning_experiments/h020/ $1

echo "h = 100"
mkdir experiments/pruning_experiments/h100/results
mkdir experiments/pruning_experiments/h100/results/tmp
python training.py experiments/pruning_experiments/h100/ $1
python evaluation.py experiments/pruning_experiments/h100/ $1

echo "h = 50"
mkdir experiments/pruning_experiments/h050/results
mkdir experiments/pruning_experiments/h050/results/tmp
python training.py experiments/pruning_experiments/h050/ $1
python evaluation.py experiments/pruning_experiments/h050/ $1

echo "h = 2"
mkdir experiments/pruning_experiments/h002/results
mkdir experiments/pruning_experiments/h002/results/tmp
python training.py experiments/pruning_experiments/h002/ $1
python evaluation.py experiments/pruning_experiments/h002/ $1

echo "h = 10"
mkdir experiments/pruning_experiments/h010/results
mkdir experiments/pruning_experiments/h010/results/tmp
python training.py experiments/pruning_experiments/h010/ $1
python evaluation.py experiments/pruning_experiments/h010/ $1
