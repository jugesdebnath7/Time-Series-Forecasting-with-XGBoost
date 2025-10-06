# inference_api.py

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from pydantic import BaseModel
import pandas as pd

# App-specific imports
from myapp.config.config_manager import ConfigManager
from myapp.utils.logger import CustomLogger
from myapp.services.inference_pipeline import InferencePipeline

# ----------------------------
# FastAPI App Initialization
# ----------------------------
app = FastAPI(
    title="Inference API",
    description="An API for running ML inference.",
    version="1.0.0"
)

# Enable CORS (optional: restrict allowed_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Config, Logger, and Pipeline
# ----------------------------
config = ConfigManager().appconfig
logger = CustomLogger(__name__).get_logger()
inference_pipeline = InferencePipeline(config=config, logger=logger)

# ----------------------------
# Response Model
# ----------------------------
class PredictionResponse(BaseModel):
    predictions: List[Dict[str, float]]

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/", tags=["Health"])
async def health_check():
    return {"message": "Inference API is running 🚀"}

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.get(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict() -> PredictionResponse:
    """
    Run the inference pipeline and return predictions as structured JSON.
    """
    try:
        logger.info("Received prediction request")

        # Run inference
        prediction_df: pd.DataFrame = inference_pipeline.run()

        # Convert DataFrame to list of dictionaries
        predictions = prediction_df.to_dict(orient="records")

        logger.info(f"Returning {len(predictions)} predictions")

        return PredictionResponse(predictions=predictions)

    except Exception as e:
        logger.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
