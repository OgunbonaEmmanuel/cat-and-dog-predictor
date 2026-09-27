from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
import numpy as np
import io
from pathlib import Path

app = FastAPI()

# Load once when the server starts
MODEL_PATH = Path(__file__).resolve().parent / "cats_vs_dogs_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    image = image.resize((150,150))

    image = np.array(image) #This converts the image into numbers

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    probability = float(prediction[0][0])

    if probability > 0.5:
        label = "Dog"
    else:
        label = "Cat"

    return {
        "prediction": label,
        "confidence": probability
    }