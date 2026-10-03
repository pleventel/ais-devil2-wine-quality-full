import logging
import os
import pickle
import sys

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "wine-quality"
VERSION_FILE = ".model-version"
OUTPUT_FILE = "wine_quality_model.pkl"
REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]


def check_env_vars() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)


def read_model_version(path: str) -> str:
    if not os.path.exists(path):
        logger.error(f"{path} not found. Create it in the repository root and put the model version in it, e.g. 1")
        sys.exit(1)
    with open(path) as f:
        version = f.read().strip()
    if not version.isdigit():
        logger.error(f"{path} must contain a single version number, but contains: '{version}'")
        sys.exit(1)
    return version


def download_model() -> None:
    check_env_vars()
    version = read_model_version(VERSION_FILE)
    model_uri = f"models:/{MODEL_NAME}/{version}"

    logger.info(f"Downloading model from registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    logger.info(f"Storing model to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    download_model()
