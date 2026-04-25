from fastapi import FastAPI, File, UploadFile
from predictor import predict_image
from llm import generate_explanation

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Pharyngitis Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_image(contents)
    explanation = generate_explanation(result["prediction"], result["confidence"])
    
    return {
        "filename": file.filename,
        "prediction": result["prediction"],
        "confidence": f"{result['confidence']}%",
        "explanation": explanation
    }