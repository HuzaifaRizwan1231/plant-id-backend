import tensorflow as tf
import numpy as np
from pathlib import Path
from src.app.utils.plant_details import class_names

class PlantIDModel:
    def __init__(self, model_path: str, model_name:str, dimensions):
        """Initialize the model by loading it from the given path."""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"{model_name} Model loaded")
        except FileNotFoundError:
            raise Exception(f"Model file not found at {model_path}")
        except Exception as e:
            raise Exception(f"Failed to load the model from {model_path}: {str(e)}")
        self.img_width = dimensions
        self.img_height = dimensions
        self.class_names = class_names

    def predict(self, image_array: np.ndarray, ensemble = False) -> dict:
        """Make a prediction on the preprocessed image array."""
        try:
            # Predict using the model
            predictions = self.model.predict(image_array)
            predicted_class = self.class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))

            if ensemble:
                return predictions

            return {"class_name": predicted_class, "confidence": confidence}
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")