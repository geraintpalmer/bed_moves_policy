alphas=("0.2" "0.4" "0.6" "0.8")
gammas=("0.6" "0.7" "0.8" "0.9")

n_cores="$1"

for alpha in "${alphas[@]}"; do
    for gamma in "${gammas[@]}"; do
        alpha_str=$(echo "$alpha" | tr -d '.')
        gamma_str=$(echo "$gamma" | tr -d '.')
        dir="experiments/param_sweep/stage1/alpha${alpha_str}_gamma${gamma_str}/"
        echo "Alpha: ${alpha}; Gamma: ${gamma}"
        python training.py "$dir" "$n_cores"
        python evaluation.py "$dir" "$n_cores"
        python analysis.py "$dir"
    done
done