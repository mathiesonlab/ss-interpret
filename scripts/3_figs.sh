set -euo pipefail

# make accuracy heatmaps (for all pops)
python compare_discs.py pred
python compare_discs.py fig

# combined correlation experiment plot (only CEU)
python correlation.py CEU exp-random 0

# linear regression experiment (only CEU)
python correlation.py CEU linreg

for pop in CEU CHB YRI
do
    # single correlation
    python correlation.py $pop corr disc_0
    python correlation.py $pop corr ${pop}_0_230410

    # forest features and single decision tree
    python dtree.py $pop forest
    python dtree.py $pop single
done
