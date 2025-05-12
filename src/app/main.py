from fastapi import FastAPI,File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from src.app.config.config import FRONTEND_URL
from src.app.models.plantid_model import PlantIDModel
from src.app.utils.image_utils import preprocess_image
from src.app.schemas.prediction import PredictionResponse

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

# Loading all the models
resnet_model = PlantIDModel("AIPlantID_ResNet50.keras", "resnet")
efficientnet_model = PlantIDModel("AIPlantID_EfficientNetB0.keras", "efficientnet")
mobilenet_model = PlantIDModel("AIPlantID_ResNet50.keras", "mobilenet")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    """Endpoint to predict the plant species from an uploaded image."""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess the image
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        
        # Make prediction
        if model_name == "resnet":
            prediction = resnet_model.predict(image_array)
        elif model_name == "efficientnet":
            prediction = efficientnet_model.predict(image_array)
        elif model_name == "mobilenet":
            prediction = mobilenet_model.predict(image_array)
        else:
            raise Exception("Invalid or no current model selected")

        # Return the response
        return PredictionResponse(class_name=prediction["class_name"], confidence=prediction["confidence"], class_details=prediction["class_details"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    


    

