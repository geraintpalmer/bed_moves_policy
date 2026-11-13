echo "Alpha=0.005, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0005/results
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0005/results/tmp
python training.py experiments/param_sweep/sweep_s2_alpha/alpha_0005/ $1
python evaluation.py experiments/param_sweep/sweep_s2_alpha/alpha_0005/ $1
rm -rf experiments/param_sweep/sweep_s2_alpha/alpha_0005/results/*.bin

echo "Alpha=0.01, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0010/results
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0010/results/tmp
python training.py experiments/param_sweep/sweep_s2_alpha/alpha_0010/ $1
python evaluation.py experiments/param_sweep/sweep_s2_alpha/alpha_0010/ $1
rm -rf experiments/param_sweep/sweep_s2_alpha/alpha_0010/results/*.bin

echo "Alpha=0.02, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0020/results
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0020/results/tmp
python training.py experiments/param_sweep/sweep_s2_alpha/alpha_0020/ $1
python evaluation.py experiments/param_sweep/sweep_s2_alpha/alpha_0020/ $1
rm -rf experiments/param_sweep/sweep_s2_alpha/alpha_0020/results/*.bin

echo "Alpha=0.05, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0050/results
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0050/results/tmp
python training.py experiments/param_sweep/sweep_s2_alpha/alpha_0050/ $1
python evaluation.py experiments/param_sweep/sweep_s2_alpha/alpha_0050/ $1
rm -rf experiments/param_sweep/sweep_s2_alpha/alpha_0050/results/*.bin

echo "Alpha=0.1, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0100/results
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0100/results/tmp
python training.py experiments/param_sweep/sweep_s2_alpha/alpha_0100/ $1
python evaluation.py experiments/param_sweep/sweep_s2_alpha/alpha_0100/ $1
rm -rf experiments/param_sweep/sweep_s2_alpha/alpha_0100/results/*.bin

echo "Alpha=0.15, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0150/results
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0150/results/tmp
python training.py experiments/param_sweep/sweep_s2_alpha/alpha_0150/ $1
python evaluation.py experiments/param_sweep/sweep_s2_alpha/alpha_0150/ $1
rm -rf experiments/param_sweep/sweep_s2_alpha/alpha_0150/results/*.bin

echo "Alpha=0.2, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0200/results
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0200/results/tmp
python training.py experiments/param_sweep/sweep_s2_alpha/alpha_0200/ $1
python evaluation.py experiments/param_sweep/sweep_s2_alpha/alpha_0200/ $1
rm -rf experiments/param_sweep/sweep_s2_alpha/alpha_0200/results/*.bin

echo "Alpha=0.25, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0250/results
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0250/results/tmp
python training.py experiments/param_sweep/sweep_s2_alpha/alpha_0250/ $1
python evaluation.py experiments/param_sweep/sweep_s2_alpha/alpha_0250/ $1
rm -rf experiments/param_sweep/sweep_s2_alpha/alpha_0250/results/*.bin

echo "Alpha=0.35, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0350/results
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0350/results/tmp
python training.py experiments/param_sweep/sweep_s2_alpha/alpha_0350/ $1
python evaluation.py experiments/param_sweep/sweep_s2_alpha/alpha_0350/ $1
rm -rf experiments/param_sweep/sweep_s2_alpha/alpha_0350/results/*.bin

echo "Alpha=0.5, Gamma=0.9995"
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0500/results
mkdir experiments/param_sweep/sweep_s2_alpha/alpha_0500/results/tmp
python training.py experiments/param_sweep/sweep_s2_alpha/alpha_0500/ $1
python evaluation.py experiments/param_sweep/sweep_s2_alpha/alpha_0500/ $1
rm -rf experiments/param_sweep/sweep_s2_alpha/alpha_0500/results/*.bin
