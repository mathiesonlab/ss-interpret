#!/bin/bash
#SBATCH --job-name=opt_discs            # Name of the job
#SBATCH --output=logs/%x_%j.out         # Stdout goes to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err          # Stderr goes to logs/jobname_jobid.err
#SBATCH --partition=dgx-b200	        # Queue to submit to
#SBATCH --ntasks=1                      # Number of tasks (usually one per process)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=32G                       # Memory allocation
#SBATCH --gpus=1
#SBATCH --time=3:00:00                  # Maximum runtime (hh:mm:ss)

# first run wandb sweep sweep.yaml
# command should be tell you a sweep code, put that here
wandb agent lzong-smathieson/ss-interpret/3sx4hiai
