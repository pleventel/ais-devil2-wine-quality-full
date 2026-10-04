import logging
import os
import pickle
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_FILE = "wine_quality_model.pkl"

# Same features, same order as CORE_FEATURES in wine_quality_training.py
FEATURES = ["alcohol", "volatile_acidity", "sulphates", "total_sulfur_dioxide", "chlorides"]

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_FILE):
        raise RuntimeError(f"{MODEL_FILE} not found. Run 'uv run download_model.py' first.")
    logger.info(f"Loading model from {MODEL_FILE}")
    with open(MODEL_FILE, "rb") as f:
        ml_models["wine_quality"] = pickle.load(f)
    logger.info("Model loaded")
    yield
    ml_models.clear()

class WineFeatures(BaseModel):
    alcohol: float = Field(ge=0, description="Alcohol content in % vol")
    volatile_acidity: float = Field(ge=0, description="Acetic acid in g/dm3")
    sulphates: float = Field(ge=0, description="Potassium sulphate in g/dm3")
    total_sulfur_dioxide: float = Field(ge=0, description="Total SO2 in mg/dm3")
    chlorides: float = Field(ge=0, description="Sodium chloride in g/dm3")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "alcohol": 10.5,
                    "volatile_acidity": 0.35,
                    "sulphates": 0.55,
                    "total_sulfur_dioxide": 115.0,
                    "chlorides": 0.05,
                }
            ]
        },
    )


class PredictionResponse(BaseModel):
    quality: float = Field(description="Predicted wine quality score")

app = FastAPI(title="Wine Quality API", version="0.1.0", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(wine: WineFeatures) -> PredictionResponse:
    X = pd.DataFrame([wine.model_dump()], columns=FEATURES)
    prediction = ml_models["wine_quality"].predict(X)[0]
    return PredictionResponse(quality=round(float(prediction), 2))
