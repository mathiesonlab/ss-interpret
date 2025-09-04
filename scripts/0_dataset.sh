set -euo pipefail

for POP in CEU CHB YRI
do
    # generate 100000 samples from pg-gan generators
    python dataset.py 230410 100000 $POP
done

for POP in CEU CHB YRI
do
    # compute summary statistics
    python compute_ss.py $POP
done

for POP in CEU CHB YRI
do
    # ad-hoc experiment: add ones
    python ad_count.py $POP
done
