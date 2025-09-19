# FixedMazeAnalyses

Need to set up mongodb and then update .env file with the following variables:
- MONGO_URI: MongoDB connection string

- MONGO_DB: Name of the MongoDB database

build the package using the following command:



Note that index for the mazes are fixed. 
 - location index: 0 - 48, where 0 is the bottom left corner and 48 is the top right corner. 
 - x, y coordinates are calculated as follows:
   - x = index % 7
   - y = index // 7
 - action index is fixed as well:
   - 0: right
   - 1: up
   - 2: left
   - 3: down
 - location-action index is calculated as follows:
   - location_action_index = action_index * 49 + location_index
 

All slurm scripts use absolute paths - change accordingly - goes under bash_scripts

what you need to get started
- create a conda environment or virtual environment
- install the correct pytorch version for your system from https://pytorch.org/get-started/locally/
- install this folder as a package using: pip install .
- install /ceph/behrens/Xiao/lowrank_lmdp as a package using: pip install /ceph/behrens/Xiao/lowrank_lmdp
- update exclude_nodes in analysis_scripts/route_inference_scripts/generate_all_lmdp_models.py and analysis_scripts/route_inference_scripts/generate_all_lmdp_hpo.py to exclude nodes that are not working for you
- set up mongodb and update .env file with own information
- update bash_scripts files with correct paths
- create folders inferred_routes/lowrank_lmdp_inferred/configs and data if they don't exist
- download the maze data from "" and put it into data folder
- download the config files and put it into inferred_routes/lowrank_lmdp_inferred/configs

If starting from scratch, including generating your own synthetic agents, then set 'fresh_run = True' in analysis_scripts/starting_script.py. Note that this produces different agents, so results will differ from those in the paper.
- run analysis_scripts/starting_script.py to generate synthetic agents, put everything into mongodb and to get relevant information for analyses - includes habits, optimal policies and pcs

Non-Markovian analysis:
- run analysis_scripts/is_it_markovian_scripts/run_analysis.py

Route inference using low-rank LMDP:
- run analysis_scripts/route_inference_scripts/generate_all_lmdp_hpo.py for hyperparameter optimization,
- run analysis_scripts/route_inference_scripts/generate_all_lmdp_models.py for route inference  # this can be run after hyperparameter optimization is done

Route navigation analysis
- run analysis_scripts/is_it_using_routes_scripts/run_analysis_pca_version.py
- run analysis_scripts/is_it_using_routes_scripts/run_analysis_lowrank_version.py # this can only be done after route inference is done

