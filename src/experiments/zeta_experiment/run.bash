echo "Zeta = 0.15"
mkdir experiments/zeta_experiment/zeta15/results
mkdir experiments/zeta_experiment/zeta15/results/tmp
python training.py experiments/zeta_experiment/zeta15/ $1
python evaluation.py experiments/zeta_experiment/zeta15/ $1

echo "Zeta = 0.85"
mkdir experiments/zeta_experiment/zeta85/results
mkdir experiments/zeta_experiment/zeta85/results/tmp
python training.py experiments/zeta_experiment/zeta85/ $1
python evaluation.py experiments/zeta_experiment/zeta85/ $1
