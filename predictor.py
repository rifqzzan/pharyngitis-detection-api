import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gdown

MODEL_PATH = "model.keras"
FILE_ID = "19mM-XQrJ_mAJjpCQwNXqx8WOYWIDk5sU"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    print("Model downloaded!")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = image.resize((224, 224))
    
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    probability = float(prediction[0][0])
    
    if probability > 0.5:
        label = "Normal"
        confidence = probability
    else:
        label = "Pharyngitis"
        confidence = 1 - probability
    
    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }
