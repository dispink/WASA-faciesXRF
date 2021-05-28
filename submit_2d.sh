#!/usr/bin/env bash

#SBATCH -J build_y
#SBATCH --qos main
#SBATCH --output=/home/users/aslee/WASA_faciesXRF/job_logs/slurm-%j.txt
#SBATCH -c 1
#SBATCH --mem=8GB
#SBATCH -t 02:00:00


#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_2d_lr.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_2d_svc.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_2d_rf.py
/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/produce_2d_evaluations.py