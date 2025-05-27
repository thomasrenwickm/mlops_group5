import argparse
import pandas as pd
import logging
import os
import yaml

from data_load.data_loader import get_data
from data_validation.data_validation import validate_schema

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


def main():
    #1. loading data
    df_raw = get_data()
    print(df_raw.head())
    print('data loading worked')
    
    #2. validating data
    config_path: str = "config.yaml"
    config = load_config(config_path)
    schema = config["data_validation"]["schema"]["columns"]
    action_on_error = config["data_validation"].get("action_on_error", "raise")
    validate_schema(df_raw, schema, action_on_error)
    print('validate schema worked')

if __name__ == "__main__":
    main()