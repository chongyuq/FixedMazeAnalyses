#!/bin/bash
#
#SBATCH --job-name=run_script
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 0-10:00
#SBATCH -o /ceph/behrens/Xiao/fixed_maze_analyses/s
# lurm_reports/slurm.%N.%j.out # STDOUT
#SBATCH -e /ceph/behrens/Xiao/fixed_maze_analyses/slurm_reports/slurm.%N.%j.err # STDERR

hostname
source ~/.bashrc
conda activate fixed_maze_analyses_conda
export PYTHONPATH=$PYTHONPATH:/ceph/behrens/Xiao/fixed_maze_analyses/src

python /ceph/behrens/Xiao/fixed_maze_analyses/analysis_scripts/route_inference_scripts/hyperparameter_search.py
#python /ceph/behrens/Xiao/fixed_maze_analyses/analysis_scripts/route_inference_scripts/generate_hmm_routes_single_subject_maze.py