"""
inference/run.py

MLflow-compatible batch inference step with Hydra config and W&B logging.
Uses the trained model and preprocessing pipeline to generate predictions
for new data. Logs prediction artifact, input hash/schema, duration,
prediction probability stats, and sample table to W&B.
"""

import sys
import logging
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
import tempfile
import pickle

import hydra
import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import yaml

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import preprocessing and feature engineering (used inline now)
from preprocess.preprocessing import transform_with_pipeline
from features.features import engineer_features

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inference")

def df_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for the inference MLflow step."""
    config_path = PROJECT_ROOT / "config.yaml"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"inference_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="inference",
            name=run_name,
            config=cfg_dict,
            tags=["inference"],
        )
        logger.info("Started WandB run: %s", run_name)

        input_path = PROJECT_ROOT / cfg.inference.input_csv
        output_path = PROJECT_ROOT / cfg.inference.output_csv

        
        logger.info("Using input CSV from config: %s", input_path)
    

        # Log input data hash and schema
        if input_path.is_file():
            in_df = pd.read_csv(input_path)
            wandb.summary["input_data_hash"] = df_hash(in_df)
            input_schema = {col: str(dtype) for col, dtype in in_df.dtypes.items()}
            wandb.summary["input_data_schema"] = input_schema

            schema_path = (
                PROJECT_ROOT / "artifacts" / f"infer_input_schema_{run.id[:8]}.json"
            )
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            with open(schema_path, "w") as f:
                json.dump(input_schema, f, indent=2)
            art = wandb.Artifact("inference_input_schema", type="schema")
            art.add_file(str(schema_path))
            run.log_artifact(art, aliases=["latest"])

            if cfg.data_load.get("log_artifacts", True):
                in_art = wandb.Artifact("predictions_input", type="predictions_input")
                in_art.add_file(str(input_path))
                run.log_artifact(in_art, aliases=["latest"])

        # Save resolved config
        resolved_cfg_path = PROJECT_ROOT / "artifacts" / f"infer_cfg_{run.id[:8]}.yaml"
        resolved_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_cfg_path, "w") as f:
            yaml.safe_dump(cfg_dict, f)

        # ------------------ Inference Logic (was in inferencer.py) ------------------
        logger.info("Loading config from %s", resolved_cfg_path)
        with open(resolved_cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["data_source"]["new_data_path"] = str(input_path)
        config["artifacts"]["inference_output"] = str(output_path)

        model_path = PROJECT_ROOT / config["artifacts"]["model"]
        pipeline_path = PROJECT_ROOT / config["artifacts"]["preprocessing_pipeline"]
        features_path = PROJECT_ROOT / config["artifacts"]["selected_features"]

        logger.info("Loading new data from %s", input_path)
        new_data = pd.read_csv(input_path)

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

        logger.info("Saving predictions to %s", output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)

        logger.info("Inference completed successfully")
        # ------------------ End of Inference Logic ------------------

        # Log duration
        wandb.summary["inference_duration_seconds"] = time.time() - run.start_time

        # Log predictions
        if output_path.is_file():
            out_df = pd.read_csv(output_path)
            wandb.log({"prediction_table": wandb.Table(dataframe=out_df)})
            wandb.summary["n_predictions"] = len(out_df)
            wandb.summary["prediction_columns"] = list(out_df.columns)

            if "prediction_proba" in out_df.columns:
                wandb.summary["prediction_proba_mean"] = float(out_df["prediction_proba"].mean())
                wandb.summary["prediction_proba_min"] = float(out_df["prediction_proba"].min())
                wandb.summary["prediction_proba_max"] = float(out_df["prediction_proba"].max())

            if cfg.data_load.get("log_artifacts", True):
                artifact = wandb.Artifact("predictions", type="predictions")
                artifact.add_file(str(output_path))
                run.log_artifact(artifact, aliases=["latest"])
                logger.info("Logged predictions artifact to WandB")

    except Exception as e:
        logger.exception("Inference failed")
        if run is not None:
            run.alert(title="Inference Step Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
