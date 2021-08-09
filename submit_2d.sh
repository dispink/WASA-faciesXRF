#!/usr/bin/env bash

#SBATCH -J grid_2d
#SBATCH --qos normal
#SBATCH --output=/home/users/aslee/WASA_faciesXRF/job_logs/slurm-%j.txt
#SBATCH -c 56
#SBATCH --mem=100GB
#SBATCH -t 2:00:00


#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_2d_lr.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_2d_svc.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_2d_rf.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/produce_2d_evaluations.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_r_2d.py
/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_r_2d_rf.py
