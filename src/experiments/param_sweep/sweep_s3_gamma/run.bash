echo "Alpha=0.5, Gamma=0.999"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099900/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099900/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_099900/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_099900/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_099900/results/*.bin

echo "Alpha=0.5, Gamma=0.9992"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099920/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099920/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_099920/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_099920/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_099920/results/*.bin

echo "Alpha=0.5, Gamma=0.99935"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099935/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099935/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_099935/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_099935/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_099935/results/*.bin

echo "Alpha=0.5, Gamma=0.99965"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099965/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099965/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_099965/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_099965/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_099965/results/*.bin

echo "Alpha=0.5, Gamma=0.9998"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099980/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099980/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_099980/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_099980/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_099980/results/*.bin

echo "Alpha=0.5, Gamma=0.9999"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099990/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099990/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_099990/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_099990/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_099990/results/*.bin
