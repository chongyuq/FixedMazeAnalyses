from datahelper.upload_data import upload_agent

if __name__ == "__main__":
    upload_agent('lmdp_agents', overwrite=True)
    upload_agent('dijkstra_agents', overwrite=True)
    upload_agent('non_markovian_agents', overwrite=True)
    upload_agent('markovian_agents', overwrite=True)
    print("All synthetic agents data uploaded successfully.")