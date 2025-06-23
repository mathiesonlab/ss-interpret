# plot loss (based on pg-gan output files) for all runs
MAIN_FOLDER="/Users/smathieson/Dropbox/ss-interpret/arch"

for POP in CEU CHB YRI
do
  for SEED in $(seq 0 19)
  do
    echo "python3 plotting/plot_loss.py -i ${MAIN_FOLDER}/out_files/${POP}_${SEED}_250331.out -o ${MAIN_FOLDER}/figures/loss/${POP}_${SEED}_250331.pdf"
    python3 plotting/plot_loss.py -i ${MAIN_FOLDER}/out_files/${POP}_${SEED}_250331.out -o ${MAIN_FOLDER}/figures/loss/${POP}_${SEED}_250331.pdf
  done
done