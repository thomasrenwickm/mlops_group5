"""
inference.py

Load trained model and preprocessing pipeline to make predictions on new data.
"""

import pickle
import logging
from pathlib import Path
import pandas as pd
import yaml

from preprocess.preprocessing import transform_with_pipeline
from features.features import engineer_features


logger = logging.getLogger(__name__)


def run_inference(input_csv: str, config_path: str, output_csv: str):
    """
    Executes the inference stage of the pipeline to generate predictions
    on new data based on pre-trained artifacts.

    Why: This function enables consistent, repeatable, and configurable
    deployment of a trained model. By externalizing inputs and outputs
    through config files and handling all steps from loading to prediction
    within one interface, we reduce the risk of human error and ensure
    reproducibility of results in production, experimentation,
    or batch use cases.
    """

    # Load config
    logger.info("Loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
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
    x_all = transform_with_pipeline(new_data, config, pipeline)
    x_processed = x_all[selected_features]

    logger.info("Loading model from %s", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info("Running prediction")
    predictions = model.predict(x_processed)

    logger.info("Saving predictions to %s", output_csv)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": predictions}).to_csv(output_csv, index=False)

    logger.info("Inference completed successfully")
