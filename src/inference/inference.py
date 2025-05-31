"""
inference.py

Batch inference entry point.
Usage:
    PYTHONPATH=. python src/inference/inference.py
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from features.features import engineer_features

logger = logging.getLogger(__name__)


def _load_pickle(path: str, label: str):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with p.open("rb") as fh:
        return pickle.load(fh)


def run_inference(input_csv: str, config_yaml: str, output_csv: str) -> None:

    # Load config
    with open(config_yaml, "r", encoding="utf-8") as f:
        config: Dict = yaml.safe_load(f)

    model_path = config["model"]["model_path"]
    pipeline_path = config["artifacts"]["preprocessing_pipeline"]
    features_path = config["artifacts"]["selected_features_path"]

    logger.info("Loading model and pipeline")
    model = _load_pickle(model_path, "trained model")
    pipeline = _load_pickle(pipeline_path, "preprocessing pipeline")

    # Load input
    logger.info("Reading input data from %s", input_csv)
    df = pd.read_csv(input_csv)

    # Feature engineering
    df_fe = engineer_features(df, config)

    # Preprocessing
    logger.info("Transforming input features")
    x_proc = pipeline.transform(df_fe)
    x_proc = pd.DataFrame(x_proc, columns=pipeline.get_feature_names_out())
    x_proc.columns = x_proc.columns.astype(str)

    # Load selected features
    selected_features = list(pd.read_json(features_path).values.flatten())
    x_final = x_proc[selected_features]

    # Predict
    logger.info("Running prediction")
    df["Predicted_SalePrice"] = model.predict(x_final)

    # Save output
    logger.info("Saving predictions to %s", output_csv)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Inference completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to input CSV")
    parser.add_argument("config_yaml", help="Path to config.yaml")
    parser.add_argument("output_csv", help="Path to save predictions")
    args = parser.parse_args()
    run_inference(args.input_csv, args.config_yaml, args.output_csv)


if __name__ == "__main__":
    main()
