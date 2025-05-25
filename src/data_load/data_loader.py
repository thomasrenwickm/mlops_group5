"""
data_loader.py

Modular data ingestion utility for Ames Housing dataset.
- Loads config from config.yaml
- Loads secrets from .env (if any)
- Supports robust error handling and logging
- Designed for MLOps-style data pipelines
"""

import os
import logging
import pandas as pd
import yaml
from dotenv import load_dotenv
# from typing import Optional

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
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        logger.info(f"Loaded data from {path}, shape={df.shape}")
        return df
    except Exception as e:
        logger.exception(f"Data loading failed. {e}")
        raise


def get_data(
    config_path: str = "config.yaml",
    env_path: str = ".env"
) -> pd.DataFrame:
    load_env(env_path)
    config = load_config(config_path)
    data_cfg = config.get("data_source", {})
    path = data_cfg.get("raw_path")

    if not path:
        raise ValueError("No 'raw_path' found in config.yaml under 'data_source'.")

    return load_data(
        path=path,
        file_type=data_cfg.get("type", "csv"),
        delimiter=data_cfg.get("delimiter", ","),
        header=data_cfg.get("header", 0),
        encoding=data_cfg.get("encoding", "utf-8"),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    try:
        df = get_data()
        print(df.head())
        logger.info(f"Ames Housing data loaded. Shape: {df.shape}")
    except Exception as e:
        logger.exception(f"Data ingestion failed {e}")
