"""
data_loader.py

Modular data ingestion utility for Ames Housing dataset.
- Loads config from config.yaml
- Loads secrets from .env (if any)
- Supports robust error handling and logging
"""

import os
import logging
import pandas as pd
import yaml
from dotenv import load_dotenv
from typing import Optional

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_env(env_path: str = ".env"):
    if os.path.isfile(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info(f"Environment variables loaded from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}")

def load_data(
    path: str,
    file_type: str = "csv",
    sheet_name: Optional[str] = None,
    delimiter: str = ",",
    header: int = 0,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    if not path or not isinstance(path, str):
        logger.error("No valid data path specified.")
        raise ValueError("Invalid path provided.")
    if not os.path.isfile(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        if file_type == "csv":
            df = pd.read_csv(path, delimiter=delimiter, header=header, encoding=encoding)
        elif file_type == "excel":
            df = pd.read_excel(path, sheet_name=sheet_name, header=header, engine="openpyxl")
            if isinstance(df, dict):
                raise ValueError(
                    "Multiple sheets detected. Please specify a single sheet_name."
                )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        logger.info(f"Loaded data from {path}, shape={df.shape}")
        return df
    except Exception as e:
        logger.exception("Data loading failed.")
        raise

# 🔁 ✅ UPDATED get_data to support raw and processed paths from config
def get_data(
    config_path: str = "config.yaml",
    env_path: str = ".env",
    data_stage: str = "raw"  # "raw" or "processed"
) -> pd.DataFrame:
    load_env(env_path)
    config = load_config(config_path)
    data_cfg = config.get("data_source", {})

    # NEW: Select correct path depending on data_stage
    if data_stage == "raw":
        path = data_cfg.get("raw_path")
    elif data_stage == "processed":
        path = data_cfg.get("processed_path")
    else:
        logger.error(f"Unknown data_stage: {data_stage}")
        raise ValueError(f"Unknown data_stage: {data_stage}")

    if not path or not isinstance(path, str):
        logger.error(f"No valid path for data_stage='{data_stage}'")
        raise ValueError(f"Missing path for data_stage='{data_stage}'")

    # Use parameters from config.yaml
    return load_data(
        path=path,
        file_type=data_cfg.get("type", "csv"),
        sheet_name=data_cfg.get("sheet_name"),
        delimiter=data_cfg.get("delimiter", ","),
        header=data_cfg.get("header", 0),
        encoding=data_cfg.get("encoding", "utf-8"),
    )

if __name__ == "__main__":

    try:
        df = get_data(data_stage="raw")  # or "processed"
        print(df.head())
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        logger.exception("Data ingestion failed.")
