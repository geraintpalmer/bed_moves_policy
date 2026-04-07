echo "Arrival multiplier: 0.25"
python3 training_varyarr.py experiments/vary_arrivals/m025
python3 evaluation_varyarr.py experiments/vary_arrivals/m025
python3 analysis.py experiments/vary_arrivals/m025

echo "Arrival multiplier: 0.5"
python3 training_varyarr.py experiments/vary_arrivals/m050
python3 evaluation_varyarr.py experiments/vary_arrivals/m050
python3 analysis.py experiments/vary_arrivals/m050

echo "Arrival multiplier: 0.75"
python3 training_varyarr.py experiments/vary_arrivals/m075
python3 evaluation_varyarr.py experiments/vary_arrivals/m075
python3 analysis.py experiments/vary_arrivals/m075

echo "Arrival multiplier: 1.0"
python3 training_varyarr.py experiments/vary_arrivals/m100
python3 evaluation_varyarr.py experiments/vary_arrivals/m100
python3 analysis.py experiments/vary_arrivals/m100

echo "Arrival multiplier: 1.25"
python3 training_varyarr.py experiments/vary_arrivals/m125
python3 evaluation_varyarr.py experiments/vary_arrivals/m125
python3 analysis.py experiments/vary_arrivals/m125

echo "Arrival multiplier: 1.5"
python3 training_varyarr.py experiments/vary_arrivals/m150
python3 evaluation_varyarr.py experiments/vary_arrivals/m150
python3 analysis.py experiments/vary_arrivals/m150

echo "Arrival multiplier: 2.0"
python3 training_varyarr.py experiments/vary_arrivals/m200
python3 evaluation_varyarr.py experiments/vary_arrivals/m200
python3 analysis.py experiments/vary_arrivals/m200

echo "Arrival multiplier: 2.5"
python3 training_varyarr.py experiments/vary_arrivals/m250
python3 evaluation_varyarr.py experiments/vary_arrivals/m250
python3 analysis.py experiments/vary_arrivals/m250

echo "Arrival multiplier: 3.0"
python3 training_varyarr.py experiments/vary_arrivals/m300
python3 evaluation_varyarr.py experiments/vary_arrivals/m300
python3 analysis.py experiments/vary_arrivals/m300

echo "Arrival multiplier: 4.0"
python3 training_varyarr.py experiments/vary_arrivals/400
python3 evaluation_varyarr.py experiments/vary_arrivals/400
python3 analysis.py experiments/vary_arrivals/400
