from fastapi import FastAPI
from fastapi import UploadFile, HTTPException
import io
from PIL import Image
import torch
from torchvision import transforms
import mlflow
import mlflow.pytorch  
from contextlib import asynccontextmanager
from pydantic import BaseModel
from src.utils import setup_logging

logger = setup_logging()


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float


ml_models={}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources here
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    ml_models["mnist_best_model"] = mlflow.pytorch.load_model("models:/mnist_best_model@champion")
    ml_models["mnist_best_model"].eval()
    logger.info("MNIST best model loaded and ready")
    yield
    # Cleanup resources here
    ml_models.clear()

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    if "mnist_best_model" not in ml_models:
        logger.warning("MNIST best model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile):
    # 1. Validate content type
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="File must be a PNG or JPEG image")
    # 2. Try to open it as an image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    tensor = transforms.ToTensor()(image).unsqueeze(0)

    with torch.no_grad():
        logits = ml_models["mnist_best_model"](tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        logger.info(f"Prediction: {prediction.item()}, Confidence: {confidence.item()}")

    return PredictionResponse(prediction=int(prediction.item()), confidence=float(confidence.item()))