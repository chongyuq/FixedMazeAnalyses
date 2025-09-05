from pymongo import MongoClient
import os
from dotenv import load_dotenv
from datahelper.upload_data import upload_behaviour


if __name__ == "__main__":
    load_dotenv()
    try:
        MONGO_URI = os.getenv("MONGODB_URI")
        MONGO_DB = os.getenv("MONGODB_DB")
    except:
        raise ValueError("MONGODB_URI and MONGODB_DB must be set in the .env file")

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    # check if collection is in mongodb
    if "behaviour" not in db.list_collection_names():
        print("Collection does not exist in MongoDB, uploading data...")
        upload_behaviour(overwrite=False)
    else:
        print("Collection already exists in MongoDB")