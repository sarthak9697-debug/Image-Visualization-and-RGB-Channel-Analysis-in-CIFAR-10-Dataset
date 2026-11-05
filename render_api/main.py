from pathlib import Path
from functools import lru_cache

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
import io

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.keras"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

app = FastAPI(title="CIFAR-10 Image Classifier API", version="1.0.0",
              description="CNN image classification API for CIFAR-10 classes.")

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@app.get("/")
def root():
    return {"message": "CIFAR-10 classifier API", "docs": "/docs", "classes": CLASS_NAMES}

@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok", "model": "loaded"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
        batch = np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, axis=0)
        probabilities = load_model().predict(batch, verbose=0)[0]
        index = int(np.argmax(probabilities))
        return {
            "predicted_class": CLASS_NAMES[index],
            "confidence": float(probabilities[index]),
            "probabilities": {CLASS_NAMES[i]: float(probabilities[i]) for i in range(10)}
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}")
