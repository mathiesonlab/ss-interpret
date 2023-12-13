TRAIN_POP=${1}  # first command line arg (i.e. CEU)
TEST_POP=${2} # second command line arg (i.e. GBR)

MAIN_FOLDER="/Users/smathieson/Dropbox/ss-interpret"
#DISC_FOLDER="/homes/smathieson/Documents/pg_gan_interpret/discriminators_og"
DATE="230410_230830"

for SEED in $(seq 0 19)
do
  DISC=${TRAIN_POP}"_"${SEED}"_"${DATE}"_finetuneAug23"
  
  # HEATMAPS (all stats)
  #echo "python3 correlation_heatmap.py ${MAIN_FOLDER}/summary_stats/stats_${TEST_POP}.npy ${MAIN_FOLDER}/hidden_pi/hidden_pi_${DISC}_${TEST_POP}.txt ${MAIN_FOLDER}/heatmaps/afterperm_${DISC}_${TEST_POP}.pdf"
  #python3 correlation_heatmap.py ${MAIN_FOLDER}/summary_stats/stats_${TEST_POP}.npy ${MAIN_FOLDER}/hidden_pi/hidden_pi_${DISC}_${TEST_POP}.txt ${MAIN_FOLDER}/heatmaps/afterperm_${DISC}_${TEST_POP}.pdf

  # HEATMAPS (per-SNP pi)
  echo "python3 correlation_heatmap.py ${MAIN_FOLDER}/hidden_pi/hidden_pi_${DISC}_${TEST_POP}.txt ${MAIN_FOLDER}/heatmaps/perSnpPi_${DISC}_${TEST_POP}.pdf"
  #python3 correlation_heatmap.py ${MAIN_FOLDER}/hidden_pi/hidden_pi_${DISC}_${TEST_POP}.txt ${MAIN_FOLDER}/heatmaps/perSnpPi_${DISC}_${TEST_POP}.pdf
done