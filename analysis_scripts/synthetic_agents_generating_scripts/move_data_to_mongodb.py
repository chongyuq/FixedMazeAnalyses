import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datahelper.upload_data import upload_agent


if __name__ == "__main__":
    load_dotenv()
    try:
        MONGO_URI = os.getenv("MONGODB_URI")
        MONGO_DB = os.getenv("MONGODB_DB")
    except:
        raise ValueError("MONGODB_URI and MONGODB_DB must be set in the .env file")

    client  = MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    # check if collections are in mongodb
    collections = db.list_collection_names()
    if "lmdp_agents" not in collections:
        upload_agent('lmdp_agents', overwrite=True)
    else:
        print("lmdp_agents collection already exists in MongoDB")
    if "dijkstra_agents" not in collections:
        upload_agent('dijkstra_agents', overwrite=True)
    else:
        print("dijkstra_agents collection already exists in MongoDB")
    if "non_markovian_agents" not in collections:
        upload_agent('non_markovian_agents', overwrite=True)
    else:
        print("non_markovian_agents collection already exists in MongoDB")
    if "markovian_agents" not in collections:
        upload_agent('markovian_agents', overwrite=True)
    else:
        print("markovian_agents collection already exists in MongoDB")
    print("All synthetic agents data uploaded successfully.")