echo "Alpha=0.1, Gamma=0.8"
mkdir experiments/param_sweep/round1/alpha01_gamma80/results
mkdir experiments/param_sweep/round1/alpha01_gamma80/results/tmp
python training.py experiments/param_sweep/round1/alpha01_gamma80/ $1
python evaluation.py experiments/param_sweep/round1/alpha01_gamma80/ $1
rm -rf experiments/param_sweep/round1/alpha01_gamma80/results/*.bin
echo "Alpha=0.1, Gamma=0.9"
mkdir experiments/param_sweep/round1/alpha01_gamma90/results
mkdir experiments/param_sweep/round1/alpha01_gamma90/results/tmp
python training.py experiments/param_sweep/round1/alpha01_gamma90/ $1
python evaluation.py experiments/param_sweep/round1/alpha01_gamma90/ $1
rm -rf experiments/param_sweep/round1/alpha01_gamma90/results/*.bin
echo "Alpha=0.1, Gamma=0.95"
mkdir experiments/param_sweep/round1/alpha01_gamma95/results
mkdir experiments/param_sweep/round1/alpha01_gamma95/results/tmp
python training.py experiments/param_sweep/round1/alpha01_gamma95/ $1
python evaluation.py experiments/param_sweep/round1/alpha01_gamma95/ $1
rm -rf experiments/param_sweep/round1/alpha01_gamma95/results/*.bin
echo "Alpha=0.1, Gamma=0.99"
mkdir experiments/param_sweep/round1/alpha01_gamma99/results
mkdir experiments/param_sweep/round1/alpha01_gamma99/results/tmp
python training.py experiments/param_sweep/round1/alpha01_gamma99/ $1
python evaluation.py experiments/param_sweep/round1/alpha01_gamma99/ $1
rm -rf experiments/param_sweep/round1/alpha01_gamma99/results/*.bin

echo "Alpha=0.2, Gamma=0.8"
mkdir experiments/param_sweep/round1/alpha02_gamma80/results
mkdir experiments/param_sweep/round1/alpha02_gamma80/results/tmp
python training.py experiments/param_sweep/round1/alpha02_gamma80/ $1
python evaluation.py experiments/param_sweep/round1/alpha02_gamma80/ $1
rm -rf experiments/param_sweep/round1/alpha02_gamma80/results/*.bin
echo "Alpha=0.2, Gamma=0.9"
mkdir experiments/param_sweep/round1/alpha02_gamma90/results
mkdir experiments/param_sweep/round1/alpha02_gamma90/results/tmp
python training.py experiments/param_sweep/round1/alpha02_gamma90/ $1
python evaluation.py experiments/param_sweep/round1/alpha02_gamma90/ $1
rm -rf experiments/param_sweep/round1/alpha02_gamma90/results/*.bin
echo "Alpha=0.2, Gamma=0.95"
mkdir experiments/param_sweep/round1/alpha02_gamma95/results
mkdir experiments/param_sweep/round1/alpha02_gamma95/results/tmp
python training.py experiments/param_sweep/round1/alpha02_gamma95/ $1
python evaluation.py experiments/param_sweep/round1/alpha02_gamma95/ $1
rm -rf experiments/param_sweep/round1/alpha02_gamma95/results/*.bin
echo "Alpha=0.2, Gamma=0.99"
mkdir experiments/param_sweep/round1/alpha02_gamma99/results
mkdir experiments/param_sweep/round1/alpha02_gamma99/results/tmp
python training.py experiments/param_sweep/round1/alpha02_gamma99/ $1
python evaluation.py experiments/param_sweep/round1/alpha02_gamma99/ $1
rm -rf experiments/param_sweep/round1/alpha02_gamma99/results/*.bin

echo "Alpha=0.3, Gamma=0.8"
mkdir experiments/param_sweep/round1/alpha03_gamma80/results
mkdir experiments/param_sweep/round1/alpha03_gamma80/results/tmp
python training.py experiments/param_sweep/round1/alpha03_gamma80/ $1
python evaluation.py experiments/param_sweep/round1/alpha03_gamma80/ $1
rm -rf experiments/param_sweep/round1/alpha03_gamma80/results/*.bin
echo "Alpha=0.3, Gamma=0.9"
mkdir experiments/param_sweep/round1/alpha03_gamma90/results
mkdir experiments/param_sweep/round1/alpha03_gamma90/results/tmp
python training.py experiments/param_sweep/round1/alpha03_gamma90/ $1
python evaluation.py experiments/param_sweep/round1/alpha03_gamma90/ $1
rm -rf experiments/param_sweep/round1/alpha03_gamma90/results/*.bin
echo "Alpha=0.3, Gamma=0.95"
mkdir experiments/param_sweep/round1/alpha03_gamma95/results
mkdir experiments/param_sweep/round1/alpha03_gamma95/results/tmp
python training.py experiments/param_sweep/round1/alpha03_gamma95/ $1
python evaluation.py experiments/param_sweep/round1/alpha03_gamma95/ $1
rm -rf experiments/param_sweep/round1/alpha03_gamma95/results/*.bin
echo "Alpha=0.3, Gamma=0.99"
mkdir experiments/param_sweep/round1/alpha03_gamma99/results
mkdir experiments/param_sweep/round1/alpha03_gamma99/results/tmp
python training.py experiments/param_sweep/round1/alpha03_gamma99/ $1
python evaluation.py experiments/param_sweep/round1/alpha03_gamma99/ $1
rm -rf experiments/param_sweep/round1/alpha03_gamma99/results/*.bin

echo "Alpha=0.4, Gamma=0.8"
mkdir experiments/param_sweep/round1/alpha04_gamma80/results
mkdir experiments/param_sweep/round1/alpha04_gamma80/results/tmp
python training.py experiments/param_sweep/round1/alpha04_gamma80/ $1
python evaluation.py experiments/param_sweep/round1/alpha04_gamma80/ $1
rm -rf experiments/param_sweep/round1/alpha04_gamma80/results/*.bin
echo "Alpha=0.4, Gamma=0.9"
mkdir experiments/param_sweep/round1/alpha04_gamma90/results
mkdir experiments/param_sweep/round1/alpha04_gamma90/results/tmp
python training.py experiments/param_sweep/round1/alpha04_gamma90/ $1
python evaluation.py experiments/param_sweep/round1/alpha04_gamma90/ $1
rm -rf experiments/param_sweep/round1/alpha04_gamma90/results/*.bin
echo "Alpha=0.4, Gamma=0.95"
mkdir experiments/param_sweep/round1/alpha04_gamma95/results
mkdir experiments/param_sweep/round1/alpha04_gamma95/results/tmp
python training.py experiments/param_sweep/round1/alpha04_gamma95/ $1
python evaluation.py experiments/param_sweep/round1/alpha04_gamma95/ $1
rm -rf experiments/param_sweep/round1/alpha04_gamma95/results/*.bin
echo "Alpha=0.4, Gamma=0.99"
mkdir experiments/param_sweep/round1/alpha04_gamma99/results
mkdir experiments/param_sweep/round1/alpha04_gamma99/results/tmp
python training.py experiments/param_sweep/round1/alpha04_gamma99/ $1
python evaluation.py experiments/param_sweep/round1/alpha04_gamma99/ $1
rm -rf experiments/param_sweep/round1/alpha04_gamma99/results/*.bin