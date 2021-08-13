#!/usr/bin/env bash

#SBATCH -J grid_roll
#SBATCH --qos normal
#SBATCH --output=/home/users/aslee/WASA_faciesXRF/job_logs/slurm-%j.txt
#SBATCH -c 56
#SBATCH --mem=100GB
#SBATCH -t 01:00:00


#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/produce_errors.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_rf.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_lr.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/produce_roll_evaluations.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/build_final_model.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/SVC_smooth_test.py
/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/grid_r_lr.py
/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/grid_r_svc.py
/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/grid_r_rf.py