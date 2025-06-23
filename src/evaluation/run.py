"""
evaluation/run.py

MLflow-compatible regression evaluation step using your original metrics.
Logs metrics and artifacts using Hydra config and Weights & Biases.
"""

import sys
import logging
import os
from datetime import datetime
from pathlib import Path
import tempfile
import pandas as pd
import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import pandas as pd


###
# Add project to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# 💡 Import your original metric function
from evaluation.evaluation import evaluate_regression

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluation")


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_name = f"evaluation_{datetime.now():%Y%m%d_%H%M%S}"

    run = wandb.init(
        project=cfg.main.WANDB_PROJECT,
        entity=cfg.main.WANDB_ENTITY,
        job_type="evaluation",
        name=run_name,
        config=cfg_dict,
        tags=["evaluation"],
    )
    logger.info("Started WandB run: %s", run_name)

    try:
        # Load test predictions and labels
        y_path = PROJECT_ROOT / "artifacts" / "y_true.csv"
        y_pred_path = PROJECT_ROOT / "artifacts" / "y_pred.csv"

        if not y_path.exists() or not y_pred_path.exists():
            raise FileNotFoundError("Missing y_true.csv or y_pred.csv in artifacts/")

        y_true = pd.read_csv(y_path).squeeze()
        y_pred = pd.read_csv(y_pred_path).squeeze()

        # Run evaluation
        metrics = evaluate_regression(y_true, y_pred)
        logger.info("Evaluation metrics: %s", metrics)

        # Log to W&B
        wandb.log(metrics)
        wandb.summary.update(metrics)

        # Save to file
        metrics_path = PROJECT_ROOT / cfg.artifacts.get("metrics_path", "models/metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            import json
            json.dump(metrics, f, indent=2)

        # Log artifact
        if cfg.data_load.get("log_artifacts", True):
            artifact = wandb.Artifact("eval_metrics", type="metrics")
            artifact.add_file(str(metrics_path))
            run.log_artifact(artifact, aliases=["latest"])
            logger.info("Logged metrics artifact to WandB")

    except Exception as e:
        logger.exception("Evaluation step failed")
        if wandb.run is not None:
            run.alert(title="Evaluation Error", text=str(e))
        sys.exit(1)
    finally:
        wandb.finish()
        logger.info("WandB run finished")


if __name__ == "__main__":
    main()
