#!/usr/bin/env bash

#SBATCH -J grid_svc
#SBATCH --qos main
#SBATCH --output=/home/users/aslee/WASA_faciesXRF/job_logs/slurm-%j.txt
#SBATCH -c 56
#SBATCH --mem=50GB
#SBATCH -t 30:00:00


#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/build_trainn_svc.py
/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_svc.py