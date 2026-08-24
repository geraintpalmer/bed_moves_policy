echo "Alpha=0.04, Gamma=0.85"
mkdir experiments/param_sweep/round2/alpha004_gamma85/results
mkdir experiments/param_sweep/round2/alpha004_gamma85/results/tmp
python training.py experiments/param_sweep/round2/alpha004_gamma85/ $1
python evaluation.py experiments/param_sweep/round2/alpha004_gamma85/ $1
rm -rf experiments/param_sweep/round2/alpha004_gamma85/results/*.bin
echo "Alpha=0.04, Gamma=0.89"
mkdir experiments/param_sweep/round2/alpha004_gamma89/results
mkdir experiments/param_sweep/round2/alpha004_gamma89/results/tmp
python training.py experiments/param_sweep/round2/alpha004_gamma89/ $1
python evaluation.py experiments/param_sweep/round2/alpha004_gamma89/ $1
rm -rf experiments/param_sweep/round2/alpha004_gamma89/results/*.bin
echo "Alpha=0.04, Gamma=0.92"
mkdir experiments/param_sweep/round2/alpha004_gamma92/results
mkdir experiments/param_sweep/round2/alpha004_gamma92/results/tmp
python training.py experiments/param_sweep/round2/alpha004_gamma92/ $1
python evaluation.py experiments/param_sweep/round2/alpha004_gamma92/ $1
rm -rf experiments/param_sweep/round2/alpha004_gamma92/results/*.bin
echo "Alpha=0.04, Gamma=0.94"
mkdir experiments/param_sweep/round2/alpha004_gamma94/results
mkdir experiments/param_sweep/round2/alpha004_gamma94/results/tmp
python training.py experiments/param_sweep/round2/alpha004_gamma94/ $1
python evaluation.py experiments/param_sweep/round2/alpha004_gamma94/ $1
rm -rf experiments/param_sweep/round2/alpha004_gamma94/results/*.bin

echo "Alpha=0.08, Gamma=0.85"
mkdir experiments/param_sweep/round2/alpha008_gamma85/results
mkdir experiments/param_sweep/round2/alpha008_gamma85/results/tmp
python training.py experiments/param_sweep/round2/alpha008_gamma85/ $1
python evaluation.py experiments/param_sweep/round2/alpha008_gamma85/ $1
rm -rf experiments/param_sweep/round2/alpha008_gamma85/results/*.bin
echo "Alpha=0.08, Gamma=0.89"
mkdir experiments/param_sweep/round2/alpha008_gamma89/results
mkdir experiments/param_sweep/round2/alpha008_gamma89/results/tmp
python training.py experiments/param_sweep/round2/alpha008_gamma89/ $1
python evaluation.py experiments/param_sweep/round2/alpha008_gamma89/ $1
rm -rf experiments/param_sweep/round2/alpha008_gamma89/results/*.bin
echo "Alpha=0.08, Gamma=0.92"
mkdir experiments/param_sweep/round2/alpha008_gamma92/results
mkdir experiments/param_sweep/round2/alpha008_gamma92/results/tmp
python training.py experiments/param_sweep/round2/alpha008_gamma92/ $1
python evaluation.py experiments/param_sweep/round2/alpha008_gamma92/ $1
rm -rf experiments/param_sweep/round2/alpha008_gamma92/results/*.bin
echo "Alpha=0.08, Gamma=0.94"
mkdir experiments/param_sweep/round2/alpha008_gamma94/results
mkdir experiments/param_sweep/round2/alpha008_gamma94/results/tmp
python training.py experiments/param_sweep/round2/alpha008_gamma94/ $1
python evaluation.py experiments/param_sweep/round2/alpha008_gamma94/ $1
rm -rf experiments/param_sweep/round2/alpha008_gamma94/results/*.bin

echo "Alpha=0.12, Gamma=0.85"
mkdir experiments/param_sweep/round2/alpha012_gamma85/results
mkdir experiments/param_sweep/round2/alpha012_gamma85/results/tmp
python training.py experiments/param_sweep/round2/alpha012_gamma85/ $1
python evaluation.py experiments/param_sweep/round2/alpha012_gamma85/ $1
rm -rf experiments/param_sweep/round2/alpha012_gamma85/results/*.bin
echo "Alpha=0.12, Gamma=0.89"
mkdir experiments/param_sweep/round2/alpha012_gamma89/results
mkdir experiments/param_sweep/round2/alpha012_gamma89/results/tmp
python training.py experiments/param_sweep/round2/alpha012_gamma89/ $1
python evaluation.py experiments/param_sweep/round2/alpha012_gamma89/ $1
rm -rf experiments/param_sweep/round2/alpha012_gamma89/results/*.bin
echo "Alpha=0.12, Gamma=0.92"
mkdir experiments/param_sweep/round2/alpha012_gamma92/results
mkdir experiments/param_sweep/round2/alpha012_gamma92/results/tmp
python training.py experiments/param_sweep/round2/alpha012_gamma92/ $1
python evaluation.py experiments/param_sweep/round2/alpha012_gamma92/ $1
rm -rf experiments/param_sweep/round2/alpha012_gamma92/results/*.bin
echo "Alpha=0.12, Gamma=0.94"
mkdir experiments/param_sweep/round2/alpha012_gamma94/results
mkdir experiments/param_sweep/round2/alpha012_gamma94/results/tmp
python training.py experiments/param_sweep/round2/alpha012_gamma94/ $1
python evaluation.py experiments/param_sweep/round2/alpha012_gamma94/ $1
rm -rf experiments/param_sweep/round2/alpha012_gamma94/results/*.bin

echo "Alpha=0.16, Gamma=0.85"
mkdir experiments/param_sweep/round2/alpha016_gamma85/results
mkdir experiments/param_sweep/round2/alpha016_gamma85/results/tmp
python training.py experiments/param_sweep/round2/alpha016_gamma85/ $1
python evaluation.py experiments/param_sweep/round2/alpha016_gamma85/ $1
rm -rf experiments/param_sweep/round2/alpha016_gamma85/results/*.bin
echo "Alpha=0.16, Gamma=0.89"
mkdir experiments/param_sweep/round2/alpha016_gamma89/results
mkdir experiments/param_sweep/round2/alpha016_gamma89/results/tmp
python training.py experiments/param_sweep/round2/alpha016_gamma89/ $1
python evaluation.py experiments/param_sweep/round2/alpha016_gamma89/ $1
rm -rf experiments/param_sweep/round2/alpha016_gamma89/results/*.bin
echo "Alpha=0.16, Gamma=0.92"
mkdir experiments/param_sweep/round2/alpha016_gamma92/results
mkdir experiments/param_sweep/round2/alpha016_gamma92/results/tmp
python training.py experiments/param_sweep/round2/alpha016_gamma92/ $1
python evaluation.py experiments/param_sweep/round2/alpha016_gamma92/ $1
rm -rf experiments/param_sweep/round2/alpha016_gamma92/results/*.bin
echo "Alpha=0.16, Gamma=0.94"
mkdir experiments/param_sweep/round2/alpha016_gamma94/results
mkdir experiments/param_sweep/round2/alpha016_gamma94/results/tmp
python training.py experiments/param_sweep/round2/alpha016_gamma94/ $1
python evaluation.py experiments/param_sweep/round2/alpha016_gamma94/ $1
rm -rf experiments/param_sweep/round2/alpha016_gamma94/results/*.bin