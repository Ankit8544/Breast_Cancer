import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
egg_path = r"C:\Users\HP\Desktop\PW Python\Breast Cancer\Neural Network Assignment\breastcancer\Lib\site-packages\breast_cancer_project-0.0.1-py3.8.egg\src"
sys.path.append(egg_path)
from logger import logging
from exception import CustomException
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer

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
