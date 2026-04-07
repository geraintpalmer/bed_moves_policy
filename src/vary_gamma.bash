echo "Discount Factor: 0.0"
python3 training.py experiments/vary_gamma/gamma00
python3 evaluation.py experiments/vary_gamma/gamma00
python3 analysis.py experiments/vary_gamma/gamma00

echo "Discount Factor: 0.75"
python3 training.py experiments/vary_gamma/gamma75
python3 evaluation.py experiments/vary_gamma/gamma75
python3 analysis.py experiments/vary_gamma/gamma75

echo "Discount Factor: 0.9"
python3 training.py experiments/vary_gamma/gamma90
python3 evaluation.py experiments/vary_gamma/gamma90
python3 analysis.py experiments/vary_gamma/gamma90

echo "Discount Factor: 0.95"
python3 training.py experiments/vary_gamma/gamma95
python3 evaluation.py experiments/vary_gamma/gamma95
python3 analysis.py experiments/vary_gamma/gamma95

echo "Discount Factor: 0.99"
python3 training.py experiments/vary_gamma/gamma99
python3 evaluation.py experiments/vary_gamma/gamma99
python3 analysis.py experiments/vary_gamma/gamma99

