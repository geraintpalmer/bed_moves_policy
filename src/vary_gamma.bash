echo "Discount Factor: 0.0"
python3 training.py experiments/vary_gamma/gamma00 $1
python3 evaluation.py experiments/vary_gamma/gamma00 $1
python3 analysis.py experiments/vary_gamma/gamma00 $1

echo "Discount Factor: 0.2"
python3 training.py experiments/vary_gamma/gamma20 $1
python3 evaluation.py experiments/vary_gamma/gamma20 $1
python3 analysis.py experiments/vary_gamma/gamma20 $1

echo "Discount Factor: 0.4"
python3 training.py experiments/vary_gamma/gamma40 $1
python3 evaluation.py experiments/vary_gamma/gamma40 $1
python3 analysis.py experiments/vary_gamma/gamma40 $1

echo "Discount Factor: 0.5"
python3 training.py experiments/vary_gamma/gamma50 $1
python3 evaluation.py experiments/vary_gamma/gamma50 $1
python3 analysis.py experiments/vary_gamma/gamma50 $1

echo "Discount Factor: 0.6"
python3 training.py experiments/vary_gamma/gamma60 $1
python3 evaluation.py experiments/vary_gamma/gamma60 $1
python3 analysis.py experiments/vary_gamma/gamma60 $1

echo "Discount Factor: 0.75"
python3 training.py experiments/vary_gamma/gamma75 $1
python3 evaluation.py experiments/vary_gamma/gamma75 $1
python3 analysis.py experiments/vary_gamma/gamma75 $1

echo "Discount Factor: 0.9"
python3 training.py experiments/vary_gamma/gamma90 $1
python3 evaluation.py experiments/vary_gamma/gamma90 $1
python3 analysis.py experiments/vary_gamma/gamma90 $1

echo "Discount Factor: 0.95"
python3 training.py experiments/vary_gamma/gamma95 $1
python3 evaluation.py experiments/vary_gamma/gamma95 $1
python3 analysis.py experiments/vary_gamma/gamma95 $1

echo "Discount Factor: 0.99"
python3 training.py experiments/vary_gamma/gamma99 $1
python3 evaluation.py experiments/vary_gamma/gamma99 $1
python3 analysis.py experiments/vary_gamma/gamma99 $1

