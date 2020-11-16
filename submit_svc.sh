#!/usr/bin/env bash

#SBATCH -J grid_svc
#SBATCH --qos main
#SBATCH --output=/home/users/aslee/WASA_faciesXRF/job_logs/slurm-%j.txt
#SBATCH --hint=multithread
#SBATCH --nodes=1 --exclusive --cpus-per-task=1 --ntasks-per-node=64
#SBATCH --mem=240GB
#SBATCH -t 24:00:00


/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_svc.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_svc_se.py
