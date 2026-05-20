#!/bin/bash -l
#SBATCH --job-name=150mM_base_seed
#SBATCH --account=zeelab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

module load gcc/13.2.0
module load cuda/12.4.1

export PATH="/gscratch/cheme/mzorman/03_misc/miniconda3/bin:$PATH"
conda activate mlp_cuda
python simulation.py -jobname ../NaCl_jobs/150mM_unbiased_base_seed -config configs/150mM_dang_unbiased_base_seed.txt -np 8

exit 
