#!/bin/bash
#
#SBATCH --job-name=run_model
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 0-0:30
#SBATCH -o /ceph/behrens/Xiao/fixed_maze_analyses/slurm_reports/slurm.%N.%j.out # STDOUT
#SBATCH -e /ceph/behrens/Xiao/fixed_maze_analyses/slurm_reports/slurm.%N.%j.err # STDERR
#SBATCH --array=230-661

config=/ceph/behrens/Xiao/fixed_maze_analyses/inferred_routes/lowrank_lmdp_inferred/configs/models_to_run.txt
dataset=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'NR==ArrayTaskID {print $1}' $config)
maze_number=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'NR==ArrayTaskID {print $2}' $config)
subject_index=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'NR==ArrayTaskID {print $3}' $config)
kfold=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'NR==ArrayTaskID {print $4}' $config)

hostname
source ~/.bashrc
conda activate fixed_maze_analyses_conda
export PYTHONPATH=$PYTHONPATH:/ceph/behrens/Xiao/fixed_maze_analyses/src

python /ceph/behrens/Xiao/fixed_maze_analyses/analysis_scripts/route_inference_scripts/generate_hmm_routes_single_subject_maze_with_optimal_configs.py --dataset=${dataset} --maze_number=${maze_number} --subject_index=${subject_index} --kfold=${kfold}