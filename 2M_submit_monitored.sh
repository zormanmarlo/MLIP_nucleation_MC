#!/bin/bash -l
#SBATCH --job-name=2M_gpu_benchmark
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

# Monitor GPU usage in background (samples every 2 seconds)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw \
    --format=csv -l 2 > gpu_monitor_${SLURM_JOBID}.csv &
MONITOR_PID=$!

# Run simulation
echo "Starting simulation at $(date)"
python simulation.py -jobname ../NaCl_jobs/2M_gpu_benchmark -config configs/2M_dang_unbiased.txt -np 1
echo "Finished simulation at $(date)"

# Stop monitoring
kill $MONITOR_PID

echo "GPU monitoring saved to: gpu_monitor_${SLURM_JOBID}.csv"
exit
