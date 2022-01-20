#!/usr/bin/env bash

#SBATCH -J build_model
#SBATCH --qos normal
#SBATCH --output=/home/users/aslee/WASA_faciesXRF/job_logs/slurm-%j.txt
#SBATCH -c 15
#SBATCH --mem=30GB
#SBATCH -t 05:00:00


#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/produce_errors.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_rf.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/grid_lr.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/produce_roll_evaluations.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/build_final_model.py
#/home/users/aslee/miniconda3/bin/python /home/users/aslee/WASA_faciesXRF/SVC_smooth_test.py
#/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/grid_r_lr.py
#/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/grid_r_svc.py
#/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/grid_r_rf.py
#/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/build_prob_svc.py
#/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/build_final_model.py
/home/users/aslee/miniconda3/envs/wasafacies/bin/python /home/users/aslee/WASA_faciesXRF/r_build_prob_svc.py