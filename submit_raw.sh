#!/usr/bin/env bash

#SBATCH -J grid_raw
#SBATCH --qos normal
#SBATCH --output=/home/users/aslee/WASA_faciesXRF/job_logs/slurm-%j.txt
#SBATCH -c 30
#SBATCH --mem=60GB
#SBATCH -t 01:00:00


#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_raw_lr.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_raw_svc.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_raw_rf.py
/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/grid_r_raw_lr.py
/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/grid_r_raw_svc.py
/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/grid_r_raw_rf.py