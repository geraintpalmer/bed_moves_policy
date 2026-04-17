#echo "Arrival multiplier: 0.2"
#python3 training_varyarr.py experiments/vary_arrivals/m020 $1
#python3 evaluation_varyarr.py experiments/vary_arrivals/m020 $1
#python3 analysis.py experiments/vary_arrivals/m020

#echo "Arrival multiplier: 0.4"
#python3 training_varyarr.py experiments/vary_arrivals/m040 $1
#python3 evaluation_varyarr.py experiments/vary_arrivals/m040 $1
#python3 analysis.py experiments/vary_arrivals/m040

#echo "Arrival multiplier: 0.6"
#python3 training_varyarr.py experiments/vary_arrivals/m060 $1
#python3 evaluation_varyarr.py experiments/vary_arrivals/m060 $1
#python3 analysis.py experiments/vary_arrivals/m060

#echo "Arrival multiplier: 0.8"
#python3 training_varyarr.py experiments/vary_arrivals/m080 $1
#python3 evaluation_varyarr.py experiments/vary_arrivals/m080 $1
#python3 analysis.py experiments/vary_arrivals/m080

#echo "Arrival multiplier: 1.0"
#python3 training_varyarr.py experiments/vary_arrivals/m100 $1
#python3 evaluation_varyarr.py experiments/vary_arrivals/m100 $1
#python3 analysis.py experiments/vary_arrivals/m100

#echo "Arrival multiplier: 1.2"
#python3 training_varyarr.py experiments/vary_arrivals/m120 $1
#python3 evaluation_varyarr.py experiments/vary_arrivals/m120 $1
#python3 analysis.py experiments/vary_arrivals/m120

echo "Arrival multiplier: 1.4"
python3 training_varyarr.py experiments/vary_arrivals/m140 $1
python3 evaluation_varyarr.py experiments/vary_arrivals/m140 $1
python3 analysis.py experiments/vary_arrivals/m140

echo "Arrival multiplier: 1.6"
python3 training_varyarr.py experiments/vary_arrivals/m160 $1
python3 evaluation_varyarr.py experiments/vary_arrivals/m160 $1
python3 analysis.py experiments/vary_arrivals/m160

echo "Arrival multiplier: 1.8"
python3 training_varyarr.py experiments/vary_arrivals/m180 $1
python3 evaluation_varyarr.py experiments/vary_arrivals/m180 $1
python3 analysis.py experiments/vary_arrivals/m180

echo "Arrival multiplier: 2.0"
python3 training_varyarr.py experiments/vary_arrivals/m200 $1
python3 evaluation_varyarr.py experiments/vary_arrivals/m200 $1
python3 analysis.py experiments/vary_arrivals/m200

echo "Arrival multiplier: 2.5"
python3 training_varyarr.py experiments/vary_arrivals/m250 $1
python3 evaluation_varyarr.py experiments/vary_arrivals/m250 $1
python3 analysis.py experiments/vary_arrivals/m250

echo "Arrival multiplier: 3.0"
python3 training_varyarr.py experiments/vary_arrivals/m300 $1
python3 evaluation_varyarr.py experiments/vary_arrivals/m300 $1
python3 analysis.py experiments/vary_arrivals/m300
