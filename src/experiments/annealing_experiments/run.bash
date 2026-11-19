echo "Linear 11 Stages"
mkdir experiments/annealing_experiments/linear11/results
mkdir experiments/annealing_experiments/linear11/results/tmp
python training.py experiments/annealing_experiments/linear11/ $1
python evaluation.py experiments/annealing_experiments/linear11/ $1

echo "Convex 11 Stages"
mkdir experiments/annealing_experiments/convex11/results
mkdir experiments/annealing_experiments/convex11/results/tmp
python training.py experiments/annealing_experiments/convex11/ $1
python evaluation.py experiments/annealing_experiments/convex11/ $1

echo "Concave 11 Stages"
mkdir experiments/annealing_experiments/concave11/results
mkdir experiments/annealing_experiments/concave11/results/tmp
python training.py experiments/annealing_experiments/concave11/ $1
python evaluation.py experiments/annealing_experiments/concave11/ $1

echo "Linear 21 Stages"
mkdir experiments/annealing_experiments/linear21/results
mkdir experiments/annealing_experiments/linear21/results/tmp
python training.py experiments/annealing_experiments/linear21/ $1
python evaluation.py experiments/annealing_experiments/linear21/ $1

echo "Convex 21 Stages"
mkdir experiments/annealing_experiments/convex21/results
mkdir experiments/annealing_experiments/convex21/results/tmp
python training.py experiments/annealing_experiments/convex21/ $1
python evaluation.py experiments/annealing_experiments/convex21/ $1

echo "Concave 21 Stages"
mkdir experiments/annealing_experiments/concave21/results
mkdir experiments/annealing_experiments/concave21/results/tmp
python training.py experiments/annealing_experiments/concave21/ $1
python evaluation.py experiments/annealing_experiments/concave21/ $1

echo "Linear 31 Stages"
mkdir experiments/annealing_experiments/linear31/results
mkdir experiments/annealing_experiments/linear31/results/tmp
python training.py experiments/annealing_experiments/linear31/ $1
python evaluation.py experiments/annealing_experiments/linear31/ $1

echo "Convex 31 Stages"
mkdir experiments/annealing_experiments/comvex31/results
mkdir experiments/annealing_experiments/comvex31/results/tmp
python training.py experiments/annealing_experiments/comvex31/ $1
python evaluation.py experiments/annealing_experiments/comvex31/ $1

echo "Concave 31 Stages"
mkdir experiments/annealing_experiments/concave31/results
mkdir experiments/annealing_experiments/concave31/results/tmp
python training.py experiments/annealing_experiments/concave31/ $1
python evaluation.py experiments/annealing_experiments/concave31/ $1

echo "Linear 41 Stages"
mkdir experiments/annealing_experiments/linear41/results
mkdir experiments/annealing_experiments/linear41/results/tmp
python training.py experiments/annealing_experiments/linear41/ $1
python evaluation.py experiments/annealing_experiments/linear41/ $1

echo "Convex 41 Stages"
mkdir experiments/annealing_experiments/convex41/results
mkdir experiments/annealing_experiments/convex41/results/tmp
python training.py experiments/annealing_experiments/convex41/ $1
python evaluation.py experiments/annealing_experiments/convex41/ $1

echo "Concave 41 Stages"
mkdir experiments/annealing_experiments/concave41/results
mkdir experiments/annealing_experiments/concave41/results/tmp
python training.py experiments/annealing_experiments/concave41/ $1
python evaluation.py experiments/annealing_experiments/concave41/ $1
