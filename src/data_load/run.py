"""
mlflow run . -P steps="data_load"

data_load/run.py

MLflow-compatible, modular data loading step with Hydra config, W&B artifact logging, and robust error handling.
"""

import sys
import logging
import hydra
import wandb
from omegaconf import DictConfig
from datetime import datetime
from data_loader import get_data
from dotenv import load_dotenv
from pathlib import Path
import subprocess

print("🔁 Pulling DVC data...")
subprocess.run(["dvc", "pull", "--force"], check=True)

# Load environment variables once at the top
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("data_load")


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config_path = PROJECT_ROOT / "config.yaml"

    output_dir = PROJECT_ROOT / cfg.data_load.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path_cfg = Path(cfg.data_source.raw_path)
    resolved_raw_path = (
        raw_path_cfg if raw_path_cfg.is_absolute() else PROJECT_ROOT / raw_path_cfg
    )
    if not resolved_raw_path.is_file():
        raise FileNotFoundError(f"Data file not found: {resolved_raw_path}") #this is causing the issue

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = resolved_raw_path.name
    run_name = f"data_load_{dt_str}_{data_file}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="data_load",
            name=run_name,
            config=dict(cfg),
            tags=["data_load", data_file]
        )
        logger.info("Started WandB run: %s", run_name)

        # ✅ FIXED: Pass a valid path to env_path (or None if not needed anymore)
        df = get_data(
            config_path=config_path,
            data_stage=cfg.data_load.data_stage,
            env_path=PROJECT_ROOT / ".env"  # Set explicitly
        )

        if df.empty:
            logger.warning("Loaded dataframe is empty: %s", resolved_raw_path)

        dup_count = df.duplicated().sum()
        if dup_count > 0:
            logger.warning(f"Duplicates found in data ({dup_count} rows).")

        # Optional logging
        if cfg.data_load.get("log_sample_artifacts", True):
            wandb.log({"sample_rows": wandb.Table(dataframe=df.head(100))})

        if cfg.data_load.get("log_summary_stats", True):
            stats_tbl = wandb.Table(dataframe=df.describe(include="all").T.reset_index())
            wandb.log({"summary_stats": stats_tbl})

        if cfg.data_load.get("log_artifacts", True):
            raw_art = wandb.Artifact("raw_data", type="dataset")
            raw_art.add_file(str(resolved_raw_path), name="raw_data.csv")
            run.log_artifact(raw_art, aliases=["latest"])
            logger.info("Logged raw data artifact to WandB")

        wandb.summary.update({
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "n_duplicates": dup_count,
            "columns": list(df.columns)
        })

    except Exception as e:
        logger.exception("Failed during data loading step")
        if run is not None:
            run.alert(title="Data Load Error", text=str(e))
        sys.exit(1)

    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
