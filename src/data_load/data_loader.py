"""
data_loader.py

Modular data ingestion utility for Ames Housing dataset.
- Loads config from config.yaml
- Loads secrets from .env (if any)
- Supports robust error handling and logging
"""

import os
import logging
from typing import Optional
import pandas as pd
import yaml
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads config file to centralize and control project behavior.

    Why: Externalizing settings avoids hardcoding parameters across modules,
    supports easier collaboration, and allows quick updates or environment
    switches without touching the core logic.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_env(env_path: str = ".env"):
    """
    Loads environment variables to separate secrets and config from code.

    Why: Helps prevent sensitive information from being hardcoded, improves
    security, and supports flexibility when deploying across environments
    (local, dev, prod) without code changes.
    """
    if os.path.isfile(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info("Environment variables loaded from %s", env_path)
    else:
        logger.warning("No .env file found at %s", env_path)


def load_data(
    path: str,
    file_type: str = "csv",
    sheet_name: Optional[str] = None,
    delimiter: str = ",",
    header: int = 0,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Centralizes the logic for reading external data into memory
    under controlled, validated, and observable conditions.

    Why: Data loading is a fragile and error-prone part of any pipeline.
    Wrapping it with validation and logging ensures early detection of issues,
    consistent behavior across file types, and easier debugging during
    development and deployment.
    """
    if not path or not isinstance(path, str):
        logger.error("No valid data path specified.")
        raise ValueError("Invalid path provided.")
    if not os.path.isfile(path):
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        if file_type == "csv":
            df = pd.read_csv(
                path, delimiter=delimiter, header=header, encoding=encoding
            )
        elif file_type == "excel":
            df = pd.read_excel(
                path, sheet_name=sheet_name, header=header, engine="openpyxl"
            )
            if isinstance(df, dict):
                raise ValueError(
                    "Multiple sheets detected. "
                    "Please specify a single sheet_name."
                )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        logger.info("Loaded data from %s, shape=%s", path, df.shape)
        return df
    except Exception:
        logger.exception("Data loading failed.")
        raise


# 🔁 ✅ UPDATED get_data to support raw and processed paths from config
def get_data(
    config_path: str = "config.yaml",
    env_path: str = ".env",
    data_stage: str = "raw"  # "raw" or "processed"
) -> pd.DataFrame:
    """
    Retrieves data from the correct source path based on pipeline stage.

    Why: Allows flexible switching between raw and processed data stages
    without changing code. Centralizing path selection through config files
    makes the pipeline easier to maintain, adapt to new stages, and deploy
    in varied environments.
    """
    load_env(env_path)
    config = load_config(config_path)
    data_cfg = config.get("data_source", {})

    # NEW: Select correct path depending on data_stage
    if data_stage == "raw":
        path = data_cfg.get("raw_path")
    elif data_stage == "processed":
        path = data_cfg.get("processed_path")
    else:
        logger.error("Unknown data_stage: %s", data_stage)
        raise ValueError(f"Unknown data_stage: {data_stage}")

    if not path or not isinstance(path, str):
        logger.error("No valid path for data_stage='%s'", data_stage)
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
        logger.info("Data loaded successfully. Shape: %s", df.shape)
    except Exception:
        logger.exception("Data ingestion failed.")
"""
data_loader.py

Modular data ingestion utility for Ames Housing dataset.
- Loads config from config.yaml
- Loads secrets from .env (if any)
- Supports robust error handling and logging
"""
from pathlib import Path
import os
import logging
from typing import Optional
import pandas as pd
import yaml
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads config file to centralize and control project behavior.

    Why: Externalizing settings avoids hardcoding parameters across modules,
    supports easier collaboration, and allows quick updates or environment
    switches without touching the core logic.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_env(env_path: str = ".env"):
    """
    Loads environment variables to separate secrets and config from code.

    Why: Helps prevent sensitive information from being hardcoded, improves
    security, and supports flexibility when deploying across environments
    (local, dev, prod) without code changes.
    """
    if os.path.isfile(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info("Environment variables loaded from %s", env_path)
    else:
        logger.warning("No .env file found at %s", env_path)


def load_data(
    path: str,
    file_type: str = "csv",
    sheet_name: Optional[str] = None,
    delimiter: str = ",",
    header: int = 0,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Centralizes the logic for reading external data into memory
    under controlled, validated, and observable conditions.

    Why: Data loading is a fragile and error-prone part of any pipeline.
    Wrapping it with validation and logging ensures early detection of issues,
    consistent behavior across file types, and easier debugging during
    development and deployment.
    """
    print('AAAAAAA')
    print(path)
    if not path or not isinstance(path, str):
        logger.error("No valid data path specified.")
        raise ValueError("Invalid path provided.")
    #if not os.path.isfile(path):
    #    logger.error("File not found: %s", path)
    #    raise FileNotFoundError(f"Data file not found: {path}")
    

    # Convert to Path and resolve relative to project root if needed
    raw_path = Path(path)
    if not raw_path.is_absolute():
        # Assume root is 2 levels up from src/data_loader.py
        project_root = Path(__file__).resolve().parents[2]
        raw_path = project_root / raw_path

    if not raw_path.exists():
        logger.error("File not found: %s", raw_path)
        raise FileNotFoundError(f"Data file not found: {raw_path}")

    #
    try:
        if file_type == "csv":
            #df = pd.read_csv(
            #    path, delimiter=delimiter, header=header, encoding=encoding
            #)
            df = pd.read_csv(
    raw_path, delimiter=delimiter, header=header, encoding=encoding
)

        elif file_type == "excel":
            #df = pd.read_excel(
            #    path, sheet_name=sheet_name, header=header, engine="openpyxl"
            #)
            df = pd.read_excel(
    raw_path, sheet_name=sheet_name, header=header, engine="openpyxl"
)
            if isinstance(df, dict):
                raise ValueError(
                    "Multiple sheets detected. "
                    "Please specify a single sheet_name."
                )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        #logger.info("Loaded data from %s, shape=%s", path, df.shape)
        logger.info("Loaded data from %s, shape=%s", raw_path, df.shape)

        return df
    except Exception:
        logger.exception("Data loading failed.")
        raise


# 🔁 ✅ UPDATED get_data to support raw and processed paths from config
def get_data(
    config_path: str = "config.yaml",
    env_path: str = ".env",
    data_stage: str = "raw"  # "raw" or "processed"
) -> pd.DataFrame:
    """
    Retrieves data from the correct source path based on pipeline stage.

    Why: Allows flexible switching between raw and processed data stages
    without changing code. Centralizing path selection through config files
    makes the pipeline easier to maintain, adapt to new stages, and deploy
    in varied environments.
    """
    load_env(env_path)
    config = load_config(config_path)
    data_cfg = config.get("data_source", {})

    # NEW: Select correct path depending on data_stage
    if data_stage == "raw":
        path = data_cfg.get("raw_path")
    elif data_stage == "processed":
        path = data_cfg.get("processed_path")
    else:
        logger.error("Unknown data_stage: %s", data_stage)
        raise ValueError(f"Unknown data_stage: {data_stage}")
    ###
    
    if not path or not isinstance(path, str):
        logger.error("No valid path for data_stage='%s'", data_stage)
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
        
        print("🔍 Current working directory:", os.getcwd())
        print("📂 Files in CWD:", os.listdir("."))

        df = get_data(data_stage="raw")  # or "processed"
        print(df.head())
        logger.info("Data loaded successfully. Shape: %s", df.shape)
    except Exception:
        logger.exception("Data ingestion failed.")