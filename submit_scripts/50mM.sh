#!/bin/bash -l
#SBATCH --job-name=50mM_CaCO3_lammps
#SBATCH --account=zeelab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:l40:1

#SBATCH --time=72:00:00
#SBATCH --mem=40gb

## SBATCH --workdir=$SLURM_SUBMIT_DIR

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

module load ompi/4.1.6-2
module load cuda/12.8.1
module load gcc/13.2.0

export PATH="/gscratch/cheme/mzorman/03_misc/miniconda3/bin:$PATH"
source /gscratch/cheme/mzorman/03_misc/miniconda3/etc/profile.d/conda.sh
conda activate test

export LD_LIBRARY_PATH=/gscratch/cheme/mzorman/03_misc/lammps-molecules/build:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/gscratch/cheme/mzorman/03_misc/lammps-molecules/python:$PYTHONPATH

mpiexec -n 1 python simulation.py -jobname 50mM_test -config configs/50mM_test.yaml

exit
