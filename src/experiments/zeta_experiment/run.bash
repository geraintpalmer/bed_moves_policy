echo "Zeta = 0.05"
mkdir experiments/zeta_experiment/zeta05/results
mkdir experiments/zeta_experiment/zeta05/results/tmp
python training.py experiments/zeta_experiment/zeta05/ $1
python evaluation.py experiments/zeta_experiment/zeta05/ $1

echo "Zeta = 0.2"
mkdir experiments/zeta_experiment/zeta20/results
mkdir experiments/zeta_experiment/zeta20/results/tmp
python training.py experiments/zeta_experiment/zeta20/ $1
python evaluation.py experiments/zeta_experiment/zeta20/ $1

echo "Zeta = 0.35"
mkdir experiments/zeta_experiment/zeta35/results
mkdir experiments/zeta_experiment/zeta35/results/tmp
python training.py experiments/zeta_experiment/zeta35/ $1
python evaluation.py experiments/zeta_experiment/zeta35/ $1

echo "Zeta = 0.65"
mkdir experiments/zeta_experiment/zeta65/results
mkdir experiments/zeta_experiment/zeta65/results/tmp
python training.py experiments/zeta_experiment/zeta65/ $1
python evaluation.py experiments/zeta_experiment/zeta65/ $1

echo "Zeta = 0.8"
mkdir experiments/zeta_experiment/zeta80/results
mkdir experiments/zeta_experiment/zeta80/results/tmp
python training.py experiments/zeta_experiment/zeta80/ $1
python evaluation.py experiments/zeta_experiment/zeta80/ $1

echo "Zeta = 0.95"
mkdir experiments/zeta_experiment/zeta95/results
mkdir experiments/zeta_experiment/zeta95/results/tmp
python training.py experiments/zeta_experiment/zeta95/ $1
python evaluation.py experiments/zeta_experiment/zeta95/ $1
