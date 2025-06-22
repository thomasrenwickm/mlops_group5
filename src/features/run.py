"""
feature_eng/run.py

Simple feature engineering step adapted for MLflow with WandB and Hydra.
"""

import sys
import logging
import os
from datetime import datetime
from pathlib import Path
import tempfile
import json

import hydra
import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# Setup paths so you can import from `src`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("feature_eng")


# ✅ Your original feature engineering logic, unchanged
def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df["total_sf"] = df["1st Flr SF"] + df["2nd Flr SF"] + df["Total Bsmt SF"]
    df["bathrooms"] = (
        df["Full Bath"] + 0.5 * df["Half Bath"]
        + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"]
    )
    df["house_age"] = df["Yr Sold"] - df["Year Built"]
    df["since_remodel"] = df["Yr Sold"] - df["Year Remod/Add"]
    # df["MS SubClass"] = df["MS SubClass"].astype(str)
    return df


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"feature_eng_{dt_str}"
    run = None

    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="feature_eng",
            name=run_name,
            config=cfg_dict,
            tags=["feature_eng"]
        )
        logger.info("Started WandB run: %s", run_name)

        # Load validated dataset from W&B
        val_art = run.use_artifact("validated_data:latest")
        with tempfile.TemporaryDirectory() as tmp_dir:
            val_path = val_art.download(root=tmp_dir)
            df = pd.read_csv(os.path.join(val_path, "validated_data.csv"))

        if df.empty:
            logger.warning("Loaded dataframe is empty.")

        # Apply your feature engineering logic
        df = engineer_features(df, cfg_dict)

        # Save full dataset
        processed_path = PROJECT_ROOT / cfg.data_source.processed_path
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        logger.info("Saved engineered data to %s", processed_path)

        # Save sample and schema
        sample_path = processed_path.parent / "engineered_sample.csv"
        df.head(50).to_csv(sample_path, index=False)
        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        schema_path = processed_path.parent / "engineered_schema.json"
        json.dump(schema, open(schema_path, "w"), indent=2)

        # Log artifacts
        if cfg.data_load.get("log_artifacts", True):
            artifact = wandb.Artifact("engineered_data", type="dataset")
            artifact.add_file(str(processed_path))
            artifact.add_file(str(sample_path))
            artifact.add_file(str(schema_path))
            run.log_artifact(artifact, aliases=["latest"])
            logger.info("Logged processed data artifact to WandB")

        if cfg.data_load.get("log_sample_artifacts", True):
            sample_tbl = wandb.Table(dataframe=df.head(50))
            wandb.log({"processed_sample_rows": sample_tbl})

        # Track metadata
        wandb.summary.update({
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "features_added": ["total_sf", "bathrooms", "house_age", "since_remodel"]
        })

    except Exception as e:
        logger.exception("Feature engineering step failed")
        if run is not None:
            run.alert(title="Feature Eng Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
