#!/usr/bin/env bash

#SBATCH -J gridsearch_lr_dask
#SBATCH --qos main
#SBATCH --output=/home/users/aslee/WASA_faciesXRF/job_logs/slurm-%j.txt
#SBATCH --hint=multithread
#SBATCH --nodes=1 --exclusive --cpus-per-task=1 --ntasks-per-node=64
#SBATCH --mem=240GB
#SBATCH -t 24:00:00


module purge
module load openmpi/4.0.4

mkdir ~/WASA_faciesXRF/job_logs/job_$SLURM_JOB_ID
cd ~/WASA_faciesXRF/job_logs/job_$SLURM_JOB_ID

mpirun /home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_lr_dask_mpi.py
