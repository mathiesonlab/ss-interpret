#set -euo pipefail

# make accuracy heatmaps
python compare_discs.py pred
#python compare_discs.py fig

# combined correlation experiment plot
python correlation.py exp-random 0

# linear regression experiment
python correlation.py linreg

# single correlation
#python correlation.py $pop corr disc_0
#python correlation.py $pop corr ${pop}_0_230410

# forest features and single decision tree
#python dtree.py $pop forest
#python dtree.py $pop single

