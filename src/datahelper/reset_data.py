import os
from pymongo import MongoClient
from datahelper.fields import COMMON_FIELDS


def drop_all_collections():
    """
    Drops all collections in the specified MongoDB database.
    :return: None
    """
    uri = os.getenv("MONGODB_URI")
    db_str = os.getenv("MONGODB_DB")
    client = MongoClient(uri)
    db = client[db_str]
    collection_names = list(COMMON_FIELDS[dataset]['collection'] for dataset in COMMON_FIELDS)
    for collection_name in collection_names:
        db[collection_name].drop()
        print(f"Dropped collection '{collection_name}'.")
    return