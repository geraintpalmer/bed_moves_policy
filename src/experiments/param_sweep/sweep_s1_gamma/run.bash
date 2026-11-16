echo "Alpha=0.2, Gamma=0.5"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0500000/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0500000/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0500000/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0500000/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0500000/results/*.bin

echo "Alpha=0.2, Gamma=0.75"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0750000/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0750000/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0750000/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0750000/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0750000/results/*.bin

echo "Alpha=0.2, Gamma=0.9"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0900000/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0900000/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0900000/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0900000/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0900000/results/*.bin

echo "Alpha=0.2, Gamma=0.95"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0950000/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0950000/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0950000/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0950000/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0950000/results/*.bin

echo "Alpha=0.2, Gamma=0.99"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0990000/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0990000/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0990000/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0990000/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0990000/results/*.bin

echo "Alpha=0.2, Gamma=0.995"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0995000/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0995000/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0995000/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0995000/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0995000/results/*.bin

echo "Alpha=0.2, Gamma=0.999"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999000/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999000/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0999000/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0999000/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0999000/results/*.bin

echo "Alpha=0.2, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999500/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999500/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0999500/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0999500/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0999500/results/*.bin

echo "Alpha=0.2, Gamma=0.9999"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999900/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999900/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0999900/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0999900/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0999900/results/*.bin

echo "Alpha=0.2, Gamma=0.99995"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999950/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999950/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0999950/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0999950/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0999950/results/*.bin

echo "Alpha=0.2, Gamma=0.99999"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999990/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999990/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0999990/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0999990/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0999990/results/*.bin

echo "Alpha=0.2, Gamma=0.999995"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999995/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999995/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0999995/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0999995/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0999995/results/*.bin
