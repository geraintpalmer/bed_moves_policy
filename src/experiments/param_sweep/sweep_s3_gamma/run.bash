echo "Alpha=0.5, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99950/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99950/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_99950/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_99950/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_99950/results/*.bin

echo "Alpha=0.5, Gamma=0.99965"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99965/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99965/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_99965/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_99965/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_99965/results/*.bin

echo "Alpha=0.5, Gamma=0.9998"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99980/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99980/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_99980/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_99980/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_99980/results/*.bin

echo "Alpha=0.5, Gamma=0.99991"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99991/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99991/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_99991/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_99991/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_99991/results/*.bin

echo "Alpha=0.5, Gamma=0.99993"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99993/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99993/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_99993/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_99993/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_99993/results/*.bin

echo "Alpha=0.5, Gamma=0.99995"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99995/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_99995/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_99995/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_99995/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_99995/results/*.bin