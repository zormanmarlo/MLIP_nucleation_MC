#!/bin/bash
#SBATCH --job-name=gen_50mer
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=10:00:00
#SBATCH --mem=10gb
# E-mail Notification, see man sbatch for options

## SBATCH --workdir=$SLURM_SUBMIT_DIR

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

#module load intel

module load gcc/13.2.0
source /gscratch/cheme/mzorman/03_misc/miniconda3/etc/profile.d/conda.sh
conda activate
python gen_US_inputs.py --shape sphere --conc 0.1 --pairs 500

exit 0
