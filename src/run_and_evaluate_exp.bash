SECONDS=0
python3 training.py experiments/$1 $2
python3 evaluation.py experiments/$1 $2
python3 analysis.py experiments/$1
duration=$SECONDS
echo "Time duration: $duration seconds"