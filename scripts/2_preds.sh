set -euo pipefail

for POP in CEU CHB YRI
do
    python compute_preds_lf.py $POP discs/$POP/disc_N.keras 64
done

for POP in CEU CHB YRI
do
    python compute_preds_lf.py $POP models/$POP/${POP}_N_230410.h5 128
done

python compute_preds_lf.py CEU discs/CEU/random-labels_N.keras 64
python compute_preds_lf.py CEU discs/CEU/random-weights_N.keras 64