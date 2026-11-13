echo "Alpha=0.2, Gamma=0.5"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_05000/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_05000/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_05000/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_05000/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_05000/results/*.bin

echo "Alpha=0.2, Gamma=0.75"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_07500/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_07500/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_07500/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_07500/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_07500/results/*.bin

echo "Alpha=0.2, Gamma=0.9"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09000/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09000/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_09000/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_09000/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_09000/results/*.bin

echo "Alpha=0.2, Gamma=0.95"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09500/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09500/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_09500/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_09500/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_09500/results/*.bin

echo "Alpha=0.2, Gamma=0.99"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09900/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09900/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_09900/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_09900/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_09900/results/*.bin

echo "Alpha=0.2, Gamma=0.995"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09950/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09950/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_09950/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_09950/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_09950/results/*.bin

echo "Alpha=0.2, Gamma=0.999"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09990/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09990/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_09990/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_09990/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_09990/results/*.bin

echo "Alpha=0.2, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09995/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09995/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_09995/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_09995/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_09995/results/*.bin

echo "Alpha=0.2, Gamma=0.9999"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09999/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_09999/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_09999/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_09999/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_09999/results/*.bin

echo "Alpha=0.2, Gamma=0.99995"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_099995/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_099995/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_099995/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_099995/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_099995/results/*.bin

echo "Alpha=0.2, Gamma=0.99999"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_099999/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_099999/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_099999/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_099999/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_099999/results/*.bin

echo "Alpha=0.2, Gamma=0.999995"
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999995/results
mkdir experiments/param_sweep/sweep_s1_gamma/gamma_0999995/results/tmp
python training.py experiments/param_sweep/sweep_s1_gamma/gamma_0999995/ $1
python evaluation.py experiments/param_sweep/sweep_s1_gamma/gamma_0999995/ $1
rm -rf experiments/param_sweep/sweep_s1_gamma/gamma_0999995/results/*.bin