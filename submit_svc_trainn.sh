#!/usr/bin/env bash

#SBATCH -J build_svc
#SBATCH --qos main
#SBATCH --output=/home/users/aslee/WASA_faciesXRF/job_logs/slurm-%j.txt
#SBATCH -c 20
#SBATCH --mem=20GB
#SBATCH -t 01:00:00


/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/build_trainn_svc.py