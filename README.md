# FixedMazeAnalyses

Need to set up mongodb and then update .env file with the following variables:
- MONGO_URI: MongoDB connection string

- MONGO_DB: Name of the MongoDB database


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
 