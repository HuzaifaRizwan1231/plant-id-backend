from fastapi import FastAPI,File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from src.app.config.config import FRONTEND_URL
from src.app.models.plantid_model import PlantIDModel
from src.app.utils.image_utils import preprocess_image
from src.app.schemas.prediction import PredictionResponse
from src.app.utils.plant_details import plant_info, class_names
import numpy as np

# Initialize FastAPI app
app = FastAPI()

origins= [
    FRONTEND_URL,
]

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

dimensions = {
    "resnet":180,
    "efficientnet": 224,
    "mobilenet":224
}

# Loading all the models
resnet_model = PlantIDModel("AIPlantID_ResNet50.keras", "resnet", dimensions["resnet"])
efficientnet_model = PlantIDModel("AIPlantID_EfficientNetB0.keras", "efficientnet", dimensions["efficientnet"])
mobilenet_model = PlantIDModel("AIPlantID_MobileNet_v2.keras", "mobilenet", dimensions["mobilenet"])

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    """Endpoint to predict the plant species from an uploaded image."""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        image_bytes = await file.read()

        # if ensemble option is selected
        if model_name == "ensemble":
            image_array_224 = preprocess_image(image_bytes, 224)
            image_array_180 = preprocess_image(image_bytes, 180)
            resnet_prediction = resnet_model.predict(image_array_180, ensemble=True)
            efficientnet_prediction = efficientnet_model.predict(image_array_224, ensemble=True)
            mobilenet_prediction = mobilenet_model.predict(image_array_224, ensemble=True)

            # Soft voting (average of predicted probabilities)
            avg_pred = (resnet_prediction + efficientnet_prediction + mobilenet_prediction) / 3

            predicted_class = class_names[np.argmax(avg_pred)]
            confidence = float(np.max(avg_pred[0]))
            class_details = plant_info[predicted_class]

             # Return the response
            return PredictionResponse(class_name=predicted_class, confidence=confidence, class_details=class_details)


        # if not ensemble, go for individual models prediction
        # preprocess the image
        image_array = preprocess_image(image_bytes, dimensions[model_name])
        
        # Make prediction
        if model_name == "resnet":
            prediction = resnet_model.predict(image_array)
        elif model_name == "efficientnet":
            prediction = efficientnet_model.predict(image_array)
        elif model_name == "mobilenet":
            prediction = mobilenet_model.predict(image_array)
        else:
            raise Exception("Invalid or no current model selected")

        # get the class details for the predicted class name
        class_details = plant_info[prediction["class_name"]]

        # Return the response
        return PredictionResponse(class_name=prediction["class_name"], confidence=prediction["confidence"], class_details=class_details)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    


    

