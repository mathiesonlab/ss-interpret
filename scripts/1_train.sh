#!/bin/bash
#SBATCH --job-name=discs                # Name of the job
#SBATCH --output=logs/%x_%j.out         # Stdout goes to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err          # Stderr goes to logs/jobname_jobid.err
#SBATCH --partition=dgx-b200	        # Queue to submit to
#SBATCH --ntasks=1                      # Number of tasks (usually one per process)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=32G                       # Memory allocation
#SBATCH --gpus=1
#SBATCH --time=3:00:00                  # Maximum runtime (hh:mm:ss)
#SBATCH --exclude=dgx018
#SBATCH --array=0-7

NUM_EPOCHS=20
LEARNING_RATE=0.0007126
BATCH_SIZE=32
FC_SIZE=64

echo "Extracting dataset"
tar -xf dataset.tar.gz -C /tmp

for POP in CEU CHB YRI; do
    echo "Starting training for population: $POP"
    python train.py --pop $POP \
                    --num_epochs $NUM_EPOCHS \
                    --learning_rate $LEARNING_RATE \
                    --batch_size $BATCH_SIZE \
                    --fc_size $FC_SIZE \
                    --seed $SLURM_ARRAY_TASK_ID
done

# remove from tmp
rm -rf /tmp/dataset-*
