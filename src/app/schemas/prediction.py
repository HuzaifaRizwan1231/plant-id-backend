from pydantic import BaseModel

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    class_details:dict