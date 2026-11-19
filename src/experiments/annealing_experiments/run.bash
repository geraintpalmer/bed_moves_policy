echo "Linear 11 Stages"
# mkdir experiments/annealing_experiments/linear11/results
# mkdir experiments/annealing_experiments/linear11/results/tmp
# python training.py experiments/annealing_experiments/linear11/ $1
# python evaluation.py experiments/annealing_experiments/linear11/ $1
python robust_evaluation.py experiments/annealing_experiments/linear11/ $1 11

echo "Convex 11 Stages"
# mkdir experiments/annealing_experiments/convex11/results
# mkdir experiments/annealing_experiments/convex11/results/tmp
# python training.py experiments/annealing_experiments/convex11/ $1
# python evaluation.py experiments/annealing_experiments/convex11/ $1
python robust_evaluation.py experiments/annealing_experiments/convex11/ $1 11

echo "Concave 11 Stages"
# mkdir experiments/annealing_experiments/concave11/results
# mkdir experiments/annealing_experiments/concave11/results/tmp
# python training.py experiments/annealing_experiments/concave11/ $1
# python evaluation.py experiments/annealing_experiments/concave11/ $1
python robust_evaluation.py experiments/annealing_experiments/concave11/ $1 11

echo "Linear 21 Stages"
# mkdir experiments/annealing_experiments/linear21/results
# mkdir experiments/annealing_experiments/linear21/results/tmp
# python training.py experiments/annealing_experiments/linear21/ $1
# python evaluation.py experiments/annealing_experiments/linear21/ $1
python robust_evaluation.py experiments/annealing_experiments/linear21/ $1 21

echo "Convex 21 Stages"
# mkdir experiments/annealing_experiments/convex21/results
# mkdir experiments/annealing_experiments/convex21/results/tmp
# python training.py experiments/annealing_experiments/convex21/ $1
# python evaluation.py experiments/annealing_experiments/convex21/ $1
python robust_evaluation.py experiments/annealing_experiments/convex21/ $1 21

echo "Concave 21 Stages"
# mkdir experiments/annealing_experiments/concave21/results
# mkdir experiments/annealing_experiments/concave21/results/tmp
# python training.py experiments/annealing_experiments/concave21/ $1
# python evaluation.py experiments/annealing_experiments/concave21/ $1
python robust_evaluation.py experiments/annealing_experiments/concave21/ $1 21

echo "Linear 31 Stages"
# mkdir experiments/annealing_experiments/linear31/results
# mkdir experiments/annealing_experiments/linear31/results/tmp
# python training.py experiments/annealing_experiments/linear31/ $1
# python evaluation.py experiments/annealing_experiments/linear31/ $1
python robust_evaluation.py experiments/annealing_experiments/linear31/ $1 31

echo "Convex 31 Stages"
# mkdir experiments/annealing_experiments/convex31/results
# mkdir experiments/annealing_experiments/convex31/results/tmp
# python training.py experiments/annealing_experiments/convex31/ $1
# python evaluation.py experiments/annealing_experiments/convex31/ $1
python robust_evaluation.py experiments/annealing_experiments/convex31/ $1 31

echo "Concave 31 Stages"
# mkdir experiments/annealing_experiments/concave31/results
# mkdir experiments/annealing_experiments/concave31/results/tmp
# python training.py experiments/annealing_experiments/concave31/ $1
# python evaluation.py experiments/annealing_experiments/concave31/ $1
python robust_evaluation.py experiments/annealing_experiments/concave31/ $1 31

echo "Linear 41 Stages"
# mkdir experiments/annealing_experiments/linear41/results
# mkdir experiments/annealing_experiments/linear41/results/tmp
# python training.py experiments/annealing_experiments/linear41/ $1
# python evaluation.py experiments/annealing_experiments/linear41/ $1
python robust_evaluation.py experiments/annealing_experiments/linear41/ $1 41

echo "Convex 41 Stages"
# mkdir experiments/annealing_experiments/convex41/results
# mkdir experiments/annealing_experiments/convex41/results/tmp
# python training.py experiments/annealing_experiments/convex41/ $1
# python evaluation.py experiments/annealing_experiments/convex41/ $1
python robust_evaluation.py experiments/annealing_experiments/convex41/ $1 41

echo "Concave 41 Stages"
# mkdir experiments/annealing_experiments/concave41/results
# mkdir experiments/annealing_experiments/concave41/results/tmp
# python training.py experiments/annealing_experiments/concave41/ $1
# python evaluation.py experiments/annealing_experiments/concave41/ $1
python robust_evaluation.py experiments/annealing_experiments/concave41/ $1 41

echo "Linear 51 Stages"
mkdir experiments/annealing_experiments/linear51/results
mkdir experiments/annealing_experiments/linear51/results/tmp
python training.py experiments/annealing_experiments/linear51/ $1
python evaluation.py experiments/annealing_experiments/linear51/ $1
python robust_evaluation.py experiments/annealing_experiments/linear51/ $1 51

echo "Convex 51 Stages"
mkdir experiments/annealing_experiments/convex51/results
mkdir experiments/annealing_experiments/convex51/results/tmp
python training.py experiments/annealing_experiments/convex51/ $1
python evaluation.py experiments/annealing_experiments/convex51/ $1
python robust_evaluation.py experiments/annealing_experiments/convex51/ $1 51

echo "Concave 51 Stages"
mkdir experiments/annealing_experiments/concave51/results
mkdir experiments/annealing_experiments/concave51/results/tmp
python training.py experiments/annealing_experiments/concave51/ $1
python evaluation.py experiments/annealing_experiments/concave51/ $1
python robust_evaluation.py experiments/annealing_experiments/concave51/ $1 51
