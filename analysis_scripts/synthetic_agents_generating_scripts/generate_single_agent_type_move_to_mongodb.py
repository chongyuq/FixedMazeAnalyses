from synthetic_data_generate.agent_generation_funcs import generate_synthetic_agents

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datahelper.upload_data import upload_agent


if __name__ == "__main__":
    agent_type = 'non_markovian_agents'
    for maze in [1, 2, 3]:
        generate_synthetic_agents(maze_number=maze, n_agents=6, agent_type=agent_type, overwrite=True)

    print(f"{agent_type} data generated successfully.")

    load_dotenv()
    try:
        MONGO_URI = os.getenv("MONGODB_URI")
        MONGO_DB = os.getenv("MONGODB_DB")
    except:
        raise ValueError("MONGODB_URI and MONGODB_DB must be set in the .env file")

    client  = MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    upload_agent(agent_type, overwrite=True)