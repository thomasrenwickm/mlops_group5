"""Orchestrates the MLOps pipeline
   to enable modular execution and debugging.
"""

import os
import sys
import argparse
import logging

from dotenv import load_dotenv

from data_load.data_loader import get_data, load_config
from data_validation.data_validation import validate_schema
from model.model import run_model_pipeline
from inference.inference import run_inference

load_dotenv()


def setup_logging(log_cfg):
    """
    Configure unified logging for console and file output.

    Why: Ensures all modules report to the same log system and
    enables debugging and reproducibility.
    """
    log_level = log_cfg.get("level", "INFO").upper()
    log_file = log_cfg.get("log_file", "logs/main.log")
    fmt = log_cfg.get(
        "format",
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=fmt,
        datefmt=datefmt,
        filename=log_file,
        filemode="a"
    )

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt, datefmt))
    logging.getLogger().addHandler(console)


def main():
    """
    CLI entry point to control which pipeline stage to execute.

    Why: Enables flexible partial execution for development,
    debugging, or deployment.
    """
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline Orchestrator"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to optional .env file"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "data", "train", "infer"],
        default="all",
        help="Pipeline stage to run"
    )
    parser.add_argument("--input_csv", help="CSV for inference input")
    parser.add_argument("--output_csv", help="CSV for inference output")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"[main] Config load failed: {exc}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config.get("logging", {}))
    logger = logging.getLogger(__name__)
    logger.info("Pipeline started | stage=%s", args.stage)

    try:
        # 1. Data Stage
        if args.stage in ("all", "data"):
            df_raw = get_data(
                config_path=args.config,
                env_path=args.env,
                data_stage="raw"
            )
            logger.info("Raw data loaded | shape=%s", df_raw.shape)

            if config.get("data_validation", {}).get("enabled", False):
                schema = config["data_validation"]["schema"]["columns"]
                action = config["data_validation"].get(
                    "action_on_error", "raise"
                )
                validate_schema(df_raw, schema, action)
                logger.info("Schema validation passed.")

        # 2. Training Stage
        if args.stage in ("all", "train"):
            if args.stage == "train":
                df_raw = get_data(
                    config_path=args.config,
                    env_path=args.env,
                    data_stage="raw"
                )
                if config.get("data_validation", {}).get("enabled", False):
                    schema = config["data_validation"]["schema"]["columns"]
                    action = config["data_validation"].get(
                        "action_on_error", "raise"
                    )
                    validate_schema(df_raw, schema, action)

            run_model_pipeline(df_raw, config)

        # 3. Inference Stage
        if args.stage == "infer":
            if not args.input_csv or not args.output_csv:
                logger.error(
                    "Inference requires --input_csv and --output_csv."
                )
                sys.exit(1)

            run_inference(
                input_csv=args.input_csv,
                config_path=args.config,
                output_csv=args.output_csv
            )

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
