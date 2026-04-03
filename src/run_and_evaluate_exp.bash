SECONDS=0
# python3 run_qlearning.py experiments/$1
python3 evaluate_qlearning.py experiments/$1
python3 analyse_evaluation.py experiments/$1
duration=$SECONDS
echo "Time duration: $duration seconds"