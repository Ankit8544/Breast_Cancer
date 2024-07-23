import os
import sys
# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    try:
        logging.info("Starting data transformation process")

        # Paths to the train and test data CSV files
        train_data_path = os.path.join('artifacts', 'train.csv')
        test_data_path = os.path.join('artifacts', 'test.csv')

        # Initialize DataTransformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data transformation completed")

        logging.info("Starting model training process")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)
        logging.info("Model training completed")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
