#!/bin/bash
#
#SBATCH --job-name=hyper
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 0-10:00
#SBATCH -o slurm_reports/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm_reports/slurm.%N.%j.err # STDERR
#SBATCH --array=3-19

config=hyperparameters/cognitive_know_where_seed_2.txt
cognitive_mod=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'NR==ArrayTaskID {print $1}' $config)
know_where_param=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'NR==ArrayTaskID {print $2}' $config)
seed=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'NR==ArrayTaskID {print $3}' $config)

hostname
source ~/.bashrc
conda activate rlworld3

python optimal_route_finder_scripts/optimal_route_find.py --test=0 --cognitive_mod=${cognitive_mod} --know_where_param=0 --seed=$((seed + 3)) --maze_id=1 --n_routes=6