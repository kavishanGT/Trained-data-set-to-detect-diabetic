from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import numpy as np
import uvicorn
from sklearn.metrics import accuracy_score

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


# Load the trained model
model = load("model.joblib")
scaler = load("scaler.joblib")

class PredictionRequest(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

class PredictionResponse(BaseModel):
    prediction: int
    accuracy: float

@app.post("/predict/")
async def predict_diabetes(request: PredictionRequest):
    try:
        # Extract features from request
        features = np.array([[
            request.Pregnancies,
            request.Glucose,
            request.BloodPressure,
            request.SkinThickness,
            request.Insulin,
            request.BMI,
            request.DiabetesPedigreeFunction,
            request.Age
        ]])
        
        # Make prediction
        std_data = scaler.transform(features)
        prediction = model.predict(std_data)[0]
        predicted_status = 'This person has diabetic' if int(prediction) == 1 else 'This person is non-diabete'
        #accuracy = accuracy_score([prediction], [request.Outcome])
        
        # Return prediction
        return {"status": predicted_status
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error predicting: " + str(e))
if __name__ == "_main_":
    uvicorn.run(app, host='localhost',port=8000)