# import_data.py
import pandas as pd
from sklearn.datasets import load_breast_cancer
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def load_data():
    # Load the dataset from sklearn
    data = load_breast_cancer()
    
    # Create a DataFrame from the dataset
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    return df

def upload_to_mongo(df, db_name, collection_name):
    # Connect to MongoDB
    uri = "mongodb+srv://kingoflovee56:KingofLove@cluster0.ou0xiys.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    
    # Access the specified database
    db = client[db_name]
    
    # Access the specified collection
    collection = db[collection_name]
    
    # Convert DataFrame to dictionary format
    data_dict = df.to_dict("records")
    
    # Insert data into MongoDB
    collection.insert_many(data_dict)
    
    print(f"Data successfully uploaded to MongoDB collection: {collection_name}")

def main():
    # Load the data
    df = load_data()
    
    # Define MongoDB database and collection names
    db_name = 'Breast_Cancer_Database'
    collection_name = 'Breast_Cancer_Data'
    
    # Upload data to MongoDB
    upload_to_mongo(df, db_name, collection_name)

if __name__ == "__main__":
    main()
