import numpy as np
import cv2
import tensorflow as tf

def preprocess_image(image_file: bytes, dimensions: int) -> np.ndarray:
    """Preprocess the uploaded image using OpenCV to match model input requirements."""
    try:
        # Decode the image from bytes
        image = cv2.imdecode(np.frombuffer(image_file, np.uint8), cv2.IMREAD_COLOR)
        
        # Check if the image was loaded successfully
        if image is None:
            raise ValueError("Could not load the uploaded image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Define target dimensions
        img_width, img_height = dimensions, dimensions
        
        # Resize the image to 
        image_resized = cv2.resize(image, (img_width, img_height))
        
        # Add batch dimension
        image_array = np.expand_dims(image_resized, axis=0)
        
        # Preprocess the image for ResNet50
        image_array = tf.keras.applications.resnet50.preprocess_input(image_array)
        
        # Verify the shape
        expected_shape = (1, dimensions, dimensions, 3)
        if image_array.shape != expected_shape:
            raise ValueError(f"Invalid image shape after preprocessing: expected {expected_shape}, got {image_array.shape}")
        
        return image_array
    
    except Exception as e:
        raise Exception(f"Failed to preprocess image: {str(e)}")