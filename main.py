from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from predictor import predict_image
from llm import generate_explanation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Pharyngitis Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), language: str = Form("id")):
    contents = await file.read()
    result = predict_image(contents)
    explanation = generate_explanation(result["prediction"], result["confidence"], language)
    
    return {
        "filename": file.filename,
        "prediction": result["prediction"],
        "confidence": f"{result['confidence']}%",
        "explanation": explanation
    }