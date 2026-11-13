echo "Alpha=0.35, Gamma=0.9992"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_035/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_035/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_035/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_035/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_035/results/*.bin

echo "Alpha=0.4, Gamma=0.9992"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_040/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_040/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_040/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_040/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_040/results/*.bin

echo "Alpha=0.45, Gamma=0.9992"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_045/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_045/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_045/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_045/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_045/results/*.bin

echo "Alpha=0.55, Gamma=0.9992"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_055/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_055/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_055/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_055/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_055/results/*.bin

echo "Alpha=0.6, Gamma=0.9992"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_060/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_060/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_060/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_060/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_060/results/*.bin

echo "Alpha=0.65, Gamma=0.9992"
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_065/results
mkdir experiments/param_sweep/sweep_s4_alpha/alpha_065/results/tmp
python training.py experiments/param_sweep/sweep_s4_alpha/alpha_065/ $1
python evaluation.py experiments/param_sweep/sweep_s4_alpha/alpha_065/ $1
rm -rf experiments/param_sweep/sweep_s4_alpha/alpha_065/results/*.bin