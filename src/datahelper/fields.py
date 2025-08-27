COMMON_FIELDS = {
    "mice_behaviour" : {
        "collection": "behaviour",
        "fields": ["time", "pos_idx", "reward_idx", "x_pos", "y_pos", "trial",
                   "action_class", "subject_ID", "day_on_maze", "maze_number",
                   "reward_cos_angle", "reward_sin_angle", "trial_phase"],
        "sort": ["subject_ID", "unique_id", "maze_number", "day_on_maze", "time"]
    },
    "lmdp_agents": {
        "collection": "lmdp_agents",
        # "collection": "synthetic_routes_planning_lmdp_cognitive_32_inverse_temp_10_action_0.1",
        "fields": ["time", "pos_idx", "reward_idx", "x_pos", "y_pos", "trial",
                   "action_class", "subject_ID", "day_on_maze", "maze_number",
                   "trial_phase"],
        "sort": ["subject_ID", "unique_id", "maze_number", "day_on_maze", "time"]
    },
    "dijkstra_agents": {
        "collection": "dijkstra_agents",
        # "collection": "synthetic_routes_planning_alpha_32_inverse_temp_0.5",
        "fields": ["time", "pos_idx", "reward_idx", "x_pos", "y_pos", "trial",
                   "action_class", "subject_ID", "day_on_maze", "maze_number",
                   "trial_phase"],
        "sort": ["subject_ID", "unique_id", "maze_number", "day_on_maze", "time"]
    },
    "non_markovian_agents": {
        "collection": "non_markovian_agents",
        # "collection": "synthetic_no_routes_optimal_spatial_straight_temp2",
        "fields": ["time", "pos_idx", "reward_idx", "x_pos", "y_pos", "trial",
                   "action_class", "subject_ID", "day_on_maze", "maze_number",
                   "trial_phase"],
        "sort": ["subject_ID", "unique_id", "maze_number", "day_on_maze", "time"]
    },
    "markovian_agents": {
        "collection": "markovian_agents",
        # "collection": "synthetic_no_routes_optimal_spatial_temp",
        "fields": ["time", "pos_idx", "reward_idx", "x_pos", "y_pos", "trial",
                   "action_class", "subject_ID", "day_on_maze", "maze_number",
                   "trial_phase"],
        "sort": ["subject_ID", "unique_id", "maze_number", "day_on_maze", "time"]
    }
}