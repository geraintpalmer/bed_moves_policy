echo "Learning Rate: 0.1"
python3 training.py experiments/vary_alpha/alpha01 $1
python3 evaluation.py experiments/vary_alpha/alpha01 $1
python3 analysis.py experiments/vary_alpha/alpha01 $1

echo "Learning Rate: 0.2"
python3 training.py experiments/vary_alpha/alpha02 $1
python3 evaluation.py experiments/vary_alpha/alpha02 $1
python3 analysis.py experiments/vary_alpha/alpha02 $1

echo "Learning Rate: 0.3"
python3 training.py experiments/vary_alpha/alpha03 $1
python3 evaluation.py experiments/vary_alpha/alpha03 $1
python3 analysis.py experiments/vary_alpha/alpha03 $1

echo "Learning Rate: 0.4"
python3 training.py experiments/vary_alpha/alpha04 $1
python3 evaluation.py experiments/vary_alpha/alpha04 $1
python3 analysis.py experiments/vary_alpha/alpha04 $1

echo "Learning Rate: 0.5"
python3 training.py experiments/vary_alpha/alpha05 $1
python3 evaluation.py experiments/vary_alpha/alpha05 $1
python3 analysis.py experiments/vary_alpha/alpha05 $1

echo "Learning Rate: 0.6"
python3 training.py experiments/vary_alpha/alpha06 $1
python3 evaluation.py experiments/vary_alpha/alpha06 $1
python3 analysis.py experiments/vary_alpha/alpha06 $1

echo "Learning Rate: 0.7"
python3 training.py experiments/vary_alpha/alpha07 $1
python3 evaluation.py experiments/vary_alpha/alpha07 $1
python3 analysis.py experiments/vary_alpha/alpha07 $1

echo "Learning Rate: 0.8"
python3 training.py experiments/vary_alpha/alpha08 $1
python3 evaluation.py experiments/vary_alpha/alpha08 $1
python3 analysis.py experiments/vary_alpha/alpha08 $1

echo "Learning Rate: 0.9"
python3 training.py experiments/vary_alpha/alpha09 $1
python3 evaluation.py experiments/vary_alpha/alpha09 $1
python3 analysis.py experiments/vary_alpha/alpha09 $1

