"""
data_validation.py

Validates the input raw dataset using the schema defined in config.yaml.
- Checks presence of required columns
- Validates column data types
- Fails fast or warns based on configuration
- Logs all validation results to file
"""

import json
import logging
from typing import Dict, Any
import os
import pandas as pd
import yaml
from pathlib import Path

# Load config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

schema = config["data_validation"]["schema"]["columns"]
action_on_error = config["data_validation"].get("action_on_error", "raise")
report_path = config["data_validation"].get(
    "report_path", "logs/validation_report.json")


logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame,
                    schema: list,
                    action: str = "raise"
                    ) -> Dict[str, Any]:
    """
    Validate a DataFrame against a given schema.

    Parameters:
        df (pd.DataFrame): Raw input dataframe to validate.
        schema (list): List of column schema definitions from config.
        action (str): What to do if validation fails ('raise' or 'warn').

    Returns:
        dict: A report dictionary with validation status and errors.
    """
    errors = []
    for col_def in schema:
        name = col_def["name"]
        expected_type = col_def["dtype"]
        required = col_def.get("required", False)

        if name not in df.columns:
            if required:
                msg = f"Missing required column: {name}"
                errors.append(msg)
                logger.error(msg)
            continue

        # Check data type
        actual_dtype = df[name].dtype
        if (
            expected_type == "int"
            and not pd.api.types.is_integer_dtype(actual_dtype)
        ):
            msg = (
                f"Column '{name}' expected int but found {actual_dtype}"
            )
            errors.append(msg)
            logger.error(msg)

        elif (
            expected_type == "float"
            and not pd.api.types.is_float_dtype(actual_dtype)
        ):
            msg = (
                f"Column '{name}' expected float but found {actual_dtype}"
            )
            errors.append(msg)
            logger.error(msg)

        elif (
            expected_type == "str"
            and not pd.api.types.is_string_dtype(actual_dtype)
        ):
            msg = (
                f"Column '{name}' expected str but found {actual_dtype}"
            )
            errors.append(msg)
            logger.error(msg)

    report = {
        "status": "fail" if errors else "pass",
        "error_count": len(errors),
        "errors": errors,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if errors:
        if action == "raise":
            raise ValueError(
                f"Data validation failed with {len(errors)} error(s). "
                f"See report at {report_path}"
                )

        elif action == "warn":
            logger.warning(
                "Validation completed with warnings. "
                "See report at %s",
                report_path)

    else:
        logger.info("Data validation passed with no errors.")

    return report


if __name__ == "__main__":
    try:
        df = pd.read_csv(config["data_source"]["raw_path"])
        validate_schema(df, schema, action_on_error)
        print('Data validation worked!')
    except Exception as e:
        logger.exception("Data validation failed: %s", e)
