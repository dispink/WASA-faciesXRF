#!/usr/bin/env bash

#SBATCH -J evaluation
#SBATCH --qos main
#SBATCH --output=/home/users/aslee/WASA_faciesXRF/job_logs/slurm-%j.txt
#SBATCH -c 16
#SBATCH --mem=50GB
#SBATCH -t 00:30:00


#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/produce_errors.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_rf.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_lr.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/produce_roll_evaluations.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/build_final_model.py
/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/SVC_smooth_test.py