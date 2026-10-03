import json
import logging
import os
import pickle
from datetime import datetime, timezone

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_FILE = "data/winequality-red.csv"
MODEL_FILE = "models/wine_quality_model.pkl"
METADATA_FILE = "models/wine_quality_model.metadata.json"

TARGET = "quality"
FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df

def train_model():
    logger.info(f"Loading data from {DATA_FILE}")
    df = load_data(DATA_FILE)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing in dataset: {missing}. Found: {list(df.columns)}")

    X = df[FEATURES]   # input: the 11 measurements
    y = df[TARGET]     # output: the quality score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("Model training completed")
