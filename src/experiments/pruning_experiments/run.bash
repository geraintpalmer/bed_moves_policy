echo "h = 4"
mkdir experiments/pruning_experiments/h004/results
mkdir experiments/pruning_experiments/h004/results/tmp
python training.py experiments/pruning_experiments/h004/ $1
python evaluation.py experiments/pruning_experiments/h004/ $1
python robust_evaluation.py experiments/pruning_experiments/h004/ $1 21

echo "h = 6"
mkdir experiments/pruning_experiments/h006/results
mkdir experiments/pruning_experiments/h006/results/tmp
python training.py experiments/pruning_experiments/h006/ $1
python evaluation.py experiments/pruning_experiments/h006/ $1
python robust_evaluation.py experiments/pruning_experiments/h006/ $1 21

echo "h = 7"
mkdir experiments/pruning_experiments/h007/results
mkdir experiments/pruning_experiments/h007/results/tmp
python training.py experiments/pruning_experiments/h007/ $1
python evaluation.py experiments/pruning_experiments/h007/ $1
python robust_evaluation.py experiments/pruning_experiments/h007/ $1 21

echo "h = 8"
mkdir experiments/pruning_experiments/h008/results
mkdir experiments/pruning_experiments/h008/results/tmp
python training.py experiments/pruning_experiments/h008/ $1
python evaluation.py experiments/pruning_experiments/h008/ $1
python robust_evaluation.py experiments/pruning_experiments/h008/ $1 21

echo "h = 12"
mkdir experiments/pruning_experiments/h012/results
mkdir experiments/pruning_experiments/h012/results/tmp
python training.py experiments/pruning_experiments/h012/ $1
python evaluation.py experiments/pruning_experiments/h012/ $1
python robust_evaluation.py experiments/pruning_experiments/h012/ $1 21


echo "h = 0"
# mkdir experiments/pruning_experiments/h000/results
# mkdir experiments/pruning_experiments/h000/results/tmp
# python training.py experiments/pruning_experiments/h000/ 5
# python evaluation.py experiments/pruning_experiments/h000/ 5
python robust_evaluation.py experiments/pruning_experiments/h000/ 5 21

echo "h = 1"
# mkdir experiments/pruning_experiments/h001/results
# mkdir experiments/pruning_experiments/h001/results/tmp
# python training.py experiments/pruning_experiments/h001/ 10
# python evaluation.py experiments/pruning_experiments/h001/ 10
python robust_evaluation.py experiments/pruning_experiments/h001/ 10 21

echo "h = 2"
# mkdir experiments/pruning_experiments/h002/results
# mkdir experiments/pruning_experiments/h002/results/tmp
# python training.py experiments/pruning_experiments/h002/ $1
# python evaluation.py experiments/pruning_experiments/h002/ $1
python robust_evaluation.py experiments/pruning_experiments/h002/ $1 21

echo "h = 3"
# mkdir experiments/pruning_experiments/h003/results
# mkdir experiments/pruning_experiments/h003/results/tmp
# python training.py experiments/pruning_experiments/h003/ $1
# python evaluation.py experiments/pruning_experiments/h003/ $1
python robust_evaluation.py experiments/pruning_experiments/h003/ $1 21

echo "h = 5"
# mkdir experiments/pruning_experiments/h005/results
# mkdir experiments/pruning_experiments/h005/results/tmp
# python training.py experiments/pruning_experiments/h005/ $1
# python evaluation.py experiments/pruning_experiments/h005/ $1
python robust_evaluation.py experiments/pruning_experiments/h005/ $1 21

echo "h = 10"
# mkdir experiments/pruning_experiments/h010/results
# mkdir experiments/pruning_experiments/h010/results/tmp
# python training.py experiments/pruning_experiments/h010/ $1
# python evaluation.py experiments/pruning_experiments/h010/ $1
python robust_evaluation.py experiments/pruning_experiments/h010/ $1 21

echo "h = 15"
# mkdir experiments/pruning_experiments/h015/results
# mkdir experiments/pruning_experiments/h015/results/tmp
# python training.py experiments/pruning_experiments/h015/ $1
# python evaluation.py experiments/pruning_experiments/h015/ $1
python robust_evaluation.py experiments/pruning_experiments/h015/ $1 21