echo "Alpha=0.4, Gamma=0.99965"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0400/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0400/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_0400/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_0400/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_0400/results/*.bin

echo "Alpha=0.433, Gamma=0.99965"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0433/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0433/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_0433/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_0433/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_0433/results/*.bin

echo "Alpha=0.466, Gamma=0.99965"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0466/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0466/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_0466/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_0466/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_0466/results/*.bin

echo "Alpha=0.533, Gamma=0.99965"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0533/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0533/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_0533/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_0533/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_0533/results/*.bin

echo "Alpha=0.566, Gamma=0.99965"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0566/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0566/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_0566/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_0566/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_0566/results/*.bin

echo "Alpha=0.6, Gamma=0.99965"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0600/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_0600/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_0600/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_0600/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_0600/results/*.bin