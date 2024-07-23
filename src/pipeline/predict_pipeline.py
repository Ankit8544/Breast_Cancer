import sys
import os
# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.error("Exception occurred during prediction: %s", str(e))
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 mean_radius: float,
                 mean_texture: float,
                 mean_perimeter: float,
                 mean_area: float,
                 mean_smoothness: float,
                 mean_compactness: float,
                 mean_concavity: float,
                 mean_concave_points: float,
                 mean_symmetry: float,
                 mean_fractal_dimension: float,
                 radius_error: float,
                 texture_error: float,
                 perimeter_error: float,
                 area_error: float,
                 smoothness_error: float,
                 compactness_error: float,
                 concavity_error: float,
                 concave_points_error: float,
                 symmetry_error: float,
                 fractal_dimension_error: float,
                 worst_radius: float,
                 worst_texture: float,
                 worst_perimeter: float,
                 worst_area: float,
                 worst_smoothness: float,
                 worst_compactness: float,
                 worst_concavity: float,
                 worst_concave_points: float,
                 worst_symmetry: float,
                 worst_fractal_dimension: float):
        
        self.mean_radius = mean_radius
        self.mean_texture = mean_texture
        self.mean_perimeter = mean_perimeter
        self.mean_area = mean_area
        self.mean_smoothness = mean_smoothness
        self.mean_compactness = mean_compactness
        self.mean_concavity = mean_concavity
        self.mean_concave_points = mean_concave_points
        self.mean_symmetry = mean_symmetry
        self.mean_fractal_dimension = mean_fractal_dimension
        self.radius_error = radius_error
        self.texture_error = texture_error
        self.perimeter_error = perimeter_error
        self.area_error = area_error
        self.smoothness_error = smoothness_error
        self.compactness_error = compactness_error
        self.concavity_error = concavity_error
        self.concave_points_error = concave_points_error
        self.symmetry_error = symmetry_error
        self.fractal_dimension_error = fractal_dimension_error
        self.worst_radius = worst_radius
        self.worst_texture = worst_texture
        self.worst_perimeter = worst_perimeter
        self.worst_area = worst_area
        self.worst_smoothness = worst_smoothness
        self.worst_compactness = worst_compactness
        self.worst_concavity = worst_concavity
        self.worst_concave_points = worst_concave_points
        self.worst_symmetry = worst_symmetry
        self.worst_fractal_dimension = worst_fractal_dimension

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'mean radius': [self.mean_radius],
                'mean texture': [self.mean_texture],
                'mean perimeter': [self.mean_perimeter],
                'mean area': [self.mean_area],
                'mean smoothness': [self.mean_smoothness],
                'mean compactness': [self.mean_compactness],
                'mean concavity': [self.mean_concavity],
                'mean concave points': [self.mean_concave_points],
                'mean symmetry': [self.mean_symmetry],
                'mean fractal dimension': [self.mean_fractal_dimension],
                'radius error': [self.radius_error],
                'texture error': [self.texture_error],
                'perimeter error': [self.perimeter_error],
                'area error': [self.area_error],
                'smoothness error': [self.smoothness_error],
                'compactness error': [self.compactness_error],
                'concavity error': [self.concavity_error],
                'concave points error': [self.concave_points_error],
                'symmetry error': [self.symmetry_error],
                'fractal dimension error': [self.fractal_dimension_error],
                'worst radius': [self.worst_radius],
                'worst texture': [self.worst_texture],
                'worst perimeter': [self.worst_perimeter],
                'worst area': [self.worst_area],
                'worst smoothness': [self.worst_smoothness],
                'worst compactness': [self.worst_compactness],
                'worst concavity': [self.worst_concavity],
                'worst concave points': [self.worst_concave_points],
                'worst symmetry': [self.worst_symmetry],
                'worst fractal dimension': [self.worst_fractal_dimension]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame created successfully')
            return df

        except Exception as e:
            logging.error('Exception occurred in get_data_as_dataframe: %s', str(e))
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Example usage
        sample_data = CustomData(
            mean_radius=14.0, mean_texture=25.0, mean_perimeter=90.0, mean_area=800.0,
            mean_smoothness=0.1, mean_compactness=0.2, mean_concavity=0.3, mean_concave_points=0.4,
            mean_symmetry=0.5, mean_fractal_dimension=0.6, radius_error=1.0, texture_error=1.1,
            perimeter_error=1.2, area_error=1.3, smoothness_error=1.4, compactness_error=1.5,
            concavity_error=1.6, concave_points_error=1.7, symmetry_error=1.8, fractal_dimension_error=1.9,
            worst_radius=15.0, worst_texture=26.0, worst_perimeter=91.0, worst_area=810.0,
            worst_smoothness=0.11, worst_compactness=0.21, worst_concavity=0.31, worst_concave_points=0.41,
            worst_symmetry=0.51, worst_fractal_dimension=0.61
        )

        df = sample_data.get_data_as_dataframe()
        print(df)

    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"Error in main: {e}")
