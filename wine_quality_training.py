import json
import logging
import os
import pickle
from datetime import datetime, timezone
import argparse
import sys

import pandas as pd
import sklearn
import mlflow
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_FILE = "data/winequality.parquet"
MODEL_FILE = "models/wine_quality_model.pkl"
METADATA_FILE = "models/wine_quality_model.metadata.json"

TARGET = "quality"
ALL_FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
    "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "ph", "sulphates", "alcohol",
]
CORE_FEATURES = ["alcohol", "volatile_acidity", "sulphates", "total_sulfur_dioxide", "chlorides"]


RANDOM_STATE = 42
TEST_SIZE = 0.2

EXPERIMENT_NAME = "wine-quality"
REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]

# One entry = one experiment setup. The name becomes the MLflow run name.
SETUPS = {
    "rf_baseline": {"model": "random_forest", "features": ALL_FEATURES,
                    "params": {"n_estimators": 200}},
    "rf_shallow": {"model": "random_forest", "features": ALL_FEATURES,
                   "params": {"n_estimators": 50, "max_depth": 5}},
    "rf_core_features": {"model": "random_forest", "features": CORE_FEATURES,
                         "params": {"n_estimators": 200}},
    "gradient_boosting": {"model": "gradient_boosting", "features": ALL_FEATURES,
                          "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3}},
    "ridge": {"model": "ridge", "features": ALL_FEATURES,
              "params": {"alpha": 1.0}},
}


def check_env_vars() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df

def build_model(model_type: str, params: dict):
    if model_type == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    if model_type == "ridge":
        # linear models need scaled inputs, so we chain a scaler and the model
        return make_pipeline(StandardScaler(), Ridge(**params))
    raise ValueError(f"Unknown model type: {model_type}")

def train_model(setup_name: str):
    check_env_vars()
    setup = SETUPS[setup_name]
    features = setup["features"]

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.autolog()

    with mlflow.start_run(run_name=setup_name) as run:
        logger.info(f"Started MLflow run '{setup_name}' ({run.info.run_id})")
        mlflow.set_tag("setup", setup_name)
        mlflow.log_param("feature_set", ",".join(features))
        mlflow.log_param("n_features", len(features))
    
    logger.info(f"Loading data from {DATA_FILE}")
    df = load_data(DATA_FILE)

    missing = [c for c in features + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing in dataset: {missing}. Found: {list(df.columns)}")

    X = df[features]
    y = df[TARGET]     # output: the quality score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")

    model = build_model(setup["model"], setup["params"])
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
    }

    print("\n=== Evaluation report (test set) ===")
    print(f"MAE  (avg. error in quality points): {metrics['mae']:.3f}")
    print(f"RMSE (punishes big misses more)    : {metrics['rmse']:.3f}")
    print(f"R2   (1.0 = perfect, 0.0 = mean)   : {metrics['r2']:.3f}")
    print("====================================\n")

    os.makedirs("models", exist_ok=True)

    logger.info(f"Storing model to: {MODEL_FILE}")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model_type": "RandomForestRegressor",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_file": DATA_FILE,
        "target": TARGET,
        "features": FEATURES,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "hyperparameters": {"n_estimators": 200, "random_state": RANDOM_STATE},
        "sklearn_version": sklearn.__version__,
        "metrics": metrics,
    }
    logger.info(f"Writing metadata to: {METADATA_FILE}")
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    train_model()
