echo "Learning Rate: 0.1"
python3 training.py experiments/vary_alpha/alpha01/
python3 evaluation.py experiments/vary_alpha/alpha01/
python3 analysis.py experiments/vary_alpha/alpha01/

echo "Learning Rate: 0.3"
python3 training.py experiments/vary_alpha/alpha03/
python3 evaluation.py experiments/vary_alpha/alpha03/
python3 analysis.py experiments/vary_alpha/alpha03/

echo "Learning Rate: 0.5"
python3 training.py experiments/vary_alpha/alpha05/
python3 evaluation.py experiments/vary_alpha/alpha05/
python3 analysis.py experiments/vary_alpha/alpha05/

echo "Learning Rate: 0.7"
python3 training.py experiments/vary_alpha/alpha07/
python3 evaluation.py experiments/vary_alpha/alpha07/
python3 analysis.py experiments/vary_alpha/alpha07/

echo "Learning Rate: 0.9"
python3 training.py experiments/vary_alpha/alpha09/
python3 evaluation.py experiments/vary_alpha/alpha09/
python3 analysis.py experiments/vary_alpha/alpha09/

