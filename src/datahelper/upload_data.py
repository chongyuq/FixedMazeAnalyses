import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import os
from datahelper.fields import COMMON_FIELDS
from typing import Optional, Dict


load_dotenv()


def upload_agent(agent_type: str, overwrite: bool = True):
    """
    Uploads agent data from CSV files to a specified MongoDB collection.
    :param agent_type: Type of agent data to upload, e.g., 'lmdp_agents', 'dijkstra_agents', 'non_markovian_agents', 'markovian_agents'.
    :param overwrite: If True, existing collection will be dropped before inserting new data. If False, data will be appended.
    :return: None
    """
    assert agent_type in COMMON_FIELDS.keys() and agent_type != 'mice_behaviour', "Agent type must be one of 'lmdp_agents', 'dijkstra_agents', 'non_markovian_agents', 'markovian_agents'."
    collection_name = COMMON_FIELDS[agent_type]['collection']
    root_dir = Path(__file__).parents[2]
    directory_path = f'{root_dir}/data/synthetic_data/{agent_type[:-7]}'
    dfs = []
    for maze_number in range(1, 4):
        file_path = f'{directory_path}/{agent_type}_maze_{maze_number}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"File {file_path} does not exist. Skipping.")
    if not dfs:
        print("No data files found to upload.")
        return
    df = pd.concat(dfs, ignore_index=True)
    upload_data(df, collection_name, overwrite)
    return


def upload_behaviour(overwrite: bool = True):
    """
    Uploads mice behaviour data from CSV files to a specified MongoDB collection.
    :param overwrite: If True, existing collection will be dropped before inserting new data. If False, data will be appended.
    :return: None
    """
    collection_name = COMMON_FIELDS['mice_behaviour']['collection']
    root_dir = Path(__file__).parents[2]
    file_path = f'{root_dir}/data/behaviour/processed_behavioural_data.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        upload_data(df, collection_name, overwrite)
    else:
        print(f"File {file_path} does not exist. Skipping.")
    return


def upload_data(df: pd.DataFrame, collection_name: str, overwrite: bool = True):
    """
    Uploads a DataFrame to a specified MongoDB collection.
    :param df: DataFrame containing the data to be uploaded.
    :param collection_name: Name of the MongoDB collection to upload the data to.
    :param overwrite: If True, existing collection will be dropped before inserting new data. If False, data will be appended.
    :return: None
    """
    uri = os.getenv("MONGODB_URI")
    db_str = os.getenv("MONGODB_DB")
    client = MongoClient(uri)
    db = client[db_str]
    # Convert DataFrame to dictionary records
    records = df.to_dict(orient='records')

    # Insert records into the specified collection
    if records:
        # check if the collection exists, if so, delete it
        if overwrite and collection_name in db.list_collection_names():
            db[collection_name].drop()
            print(f"Dropped existing collection '{collection_name}'.")
            db[collection_name].insert_many(records)
            print(f"Newly inserted {len(records)} records into the '{collection_name}' collection.")
        elif not overwrite and collection_name in db.list_collection_names():
            db[collection_name].insert_many(records)
            print(f"Appended {len(records)} records into the existing '{collection_name}' collection.")
        elif collection_name not in db.list_collection_names():
            db[collection_name].insert_many(records)
            print(f"Created and inserted {len(records)} records into the new '{collection_name}' collection.")
    else:
        print("No records to insert.")
    return