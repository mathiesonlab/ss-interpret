# dir
DIR=$(dirname "$0")

# input data
TRAIN_POP="CEU"
TEST_POP="GBR"
GENOME="${DIR}/../data/genomes/${TEST_POP}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5"
MASK="${DIR}/../data/genomes/20120824_strict_mask.bed"

# models
MODEL="${DIR}/../models/${TRAIN_POP}/"

# output
OUTPUT="${DIR}/../results/"

# specific suffix for the model
SUFFIX="230410.h5"
SIZE=128

# predictions
for i in {4..19}
do
    echo "Running predictions for model[${TRAIN_POP}/${SUFFIX}/${i}] on data[${TEST_POP}]"
    python3 "${DIR}/../genome_disc.py" ${GENOME} ${MASK} ${MODEL} ${OUTPUT} ${SUFFIX} ${i} ${SIZE}
done
