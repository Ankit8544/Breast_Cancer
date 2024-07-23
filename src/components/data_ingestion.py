import os
import sys
import pymongo
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self, connection_string: str, database_name: str, collection_name: str):
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.ingestion_config = DataIngestionConfig()

    def load_data_from_mongodb(self) -> pd.DataFrame:
        """Load data from MongoDB into a pandas DataFrame."""
        logging.info('Connecting to MongoDB')
        try:
            client = pymongo.MongoClient(self.connection_string, tls=True)
            db = client[self.database_name]
            collection = db[self.collection_name]

            logging.info(f'Querying data from {self.collection_name}')
            data = collection.find()
            data_list = list(data)
            df = pd.DataFrame(data_list)

            logging.info('Closing MongoDB connection')
            client.close()

            return df
        except pymongo.errors.ServerSelectionTimeoutError as err:
            logging.error(f"Server selection timeout error: {err}")
            raise CustomException(f"Server selection timeout error: {err}", sys)
        except Exception as e:
            logging.error(f"General error: {e}")
            raise CustomException(f"General error: {e}", sys)

    def initiate_data_ingestion(self) -> str:
        """Initiate data ingestion process and save data to a CSV file."""
        logging.info('Data Ingestion process starts')
        try:
            df = self.load_data_from_mongodb()
            logging.info('Data loaded into DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f'Data saved to {self.ingestion_config.raw_data_path}')
            logging.info("Train test split")
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            logging.error('Error occurred in data ingestion')
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Define MongoDB connection details
        connection_string = "mongodb+srv://kingoflovee56:KingofLove@cluster0.ou0xiys.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        database_name = "Breast_Cancer_Database"
        collection_name = "Breast_Cancer_Data"

        if not all([connection_string, database_name, collection_name]):
            raise ValueError("MongoDB connection details are not set. Please check your environment variables.")

        data_ingestion = DataIngestion(connection_string, database_name, collection_name)
        raw_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f'Data ingestion completed. Raw data saved at: {raw_data_path}')
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"Error in main: {e}")

