"""
inference.py

Load trained model and preprocessing pipeline to make predictions on new data.
"""

import pandas as pd
import pickle
import logging
import yaml
import os

from preprocess.preprocessing import transform_with_pipeline
from features.features import engineer_features


logger = logging.getLogger(__name__)


def run_inference(input_csv: str, config_path: str, output_csv: str):
    from pathlib import Path

    # Load config
    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["data_source"]["new_data_path"] = input_csv
    config["artifacts"]["inference_output"] = output_csv

    model_path = config["artifacts"]["model"]
    pipeline_path = config["artifacts"]["preprocessing_pipeline"]
    features_path = config["artifacts"]["selected_features"]

    logger.info("Loading new data from %s", input_csv)
    new_data = pd.read_csv(input_csv)

    # Feature engineering (if used)
    logger.info("Running feature engineering")
    new_data = engineer_features(new_data, config)

    logger.info("Loading pipeline from %s", pipeline_path)
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    logger.info("Loading selected features from %s", features_path)
    selected_features = pd.read_json(features_path, typ="series").tolist()

    logger.info("Applying preprocessing pipeline")
    X_all = transform_with_pipeline(new_data, config, pipeline)
    X_processed = X_all[selected_features]

    logger.info("Loading model from %s", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info("Running prediction")
    predictions = model.predict(X_processed)

    logger.info("Saving predictions to %s", output_csv)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": predictions}).to_csv(output_csv, index=False)

    logger.info("Inference completed successfully")
