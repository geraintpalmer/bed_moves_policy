#echo "Discount Factor: 0.0"
#python3 training.py experiments/vary_gamma/gamma00
#python3 evaluation.py experiments/vary_gamma/gamma00
#python3 analysis.py experiments/vary_gamma/gamma00

echo "Discount Factor: 0.2"
python3 training.py experiments/vary_gamma/gamma02
python3 evaluation.py experiments/vary_gamma/gamma02
python3 analysis.py experiments/vary_gamma/gamma02

echo "Discount Factor: 0.4"
python3 training.py experiments/vary_gamma/gamma04
python3 evaluation.py experiments/vary_gamma/gamma04
python3 analysis.py experiments/vary_gamma/gamma04

echo "Discount Factor: 0.5"
python3 training.py experiments/vary_gamma/gamma05
python3 evaluation.py experiments/vary_gamma/gamma05
python3 analysis.py experiments/vary_gamma/gamma05

echo "Discount Factor: 0.6"
python3 training.py experiments/vary_gamma/gamma06
python3 evaluation.py experiments/vary_gamma/gamma06
python3 analysis.py experiments/vary_gamma/gamma06

#echo "Discount Factor: 0.75"
#python3 training.py experiments/vary_gamma/gamma75
#python3 evaluation.py experiments/vary_gamma/gamma75
#python3 analysis.py experiments/vary_gamma/gamma75
#
#echo "Discount Factor: 0.9"
#python3 training.py experiments/vary_gamma/gamma90
#python3 evaluation.py experiments/vary_gamma/gamma90
#python3 analysis.py experiments/vary_gamma/gamma90
#
#echo "Discount Factor: 0.95"
#python3 training.py experiments/vary_gamma/gamma95
#python3 evaluation.py experiments/vary_gamma/gamma95
#python3 analysis.py experiments/vary_gamma/gamma95
#
#echo "Discount Factor: 0.99"
#python3 training.py experiments/vary_gamma/gamma99
#python3 evaluation.py experiments/vary_gamma/gamma99
#python3 analysis.py experiments/vary_gamma/gamma99

