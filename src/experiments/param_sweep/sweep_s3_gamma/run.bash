echo "Alpha=0.5, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099950/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099950/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_099950/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_099950/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_099950/results/*.bin

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

echo "Alpha=0.5, Gamma=0.99991"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099991/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099991/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_099991/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_099991/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_099991/results/*.bin

echo "Alpha=0.5, Gamma=0.99993"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099993/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099993/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_099993/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_099993/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_099993/results/*.bin

echo "Alpha=0.5, Gamma=0.99995"
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099995/results
mkdir experiments/param_sweep/sweep_s3_gamma/gamma_099995/results/tmp
python training.py experiments/param_sweep/sweep_s3_gamma/gamma_099995/ $1
python evaluation.py experiments/param_sweep/sweep_s3_gamma/gamma_099995/ $1
rm -rf experiments/param_sweep/sweep_s3_gamma/gamma_099995/results/*.bin