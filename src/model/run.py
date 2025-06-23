"""
model/run.py

MLflow-compatible model-training step with Hydra config and W&B logging.
Trains the model, logs metrics, schema, hashes and all artifacts needed
for reproducibility and auditability.
"""

import sys, logging, hashlib, json, os
from datetime import datetime
from pathlib import Path
import tempfile
import hydra, wandb, pandas as pd
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.model import run_model_pipeline
from evaluation.evaluation import evaluate_regression

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("model")

def df_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of a DataFrame (values + index)."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

#@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
###
@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)


###
def main(cfg: DictConfig) -> None:
    # 👇 ADD THIS EXACT LINE HERE
    os.chdir(PROJECT_ROOT)
    ###
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_name = f"model_{datetime.now():%Y%m%d_%H%M%S}"

    run = wandb.init(
        project=cfg.main.WANDB_PROJECT,
        entity=cfg.main.WANDB_ENTITY,
        job_type="model",
        name=run_name,
        config=cfg_dict,
        tags=["model"],
    )
    logger.info("Started WandB run: %s", run_name)

    try:
        data_art = run.use_artifact("validated_data:latest")
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = data_art.download(root=tmp_dir)

            data_file = os.path.join(data_path, "validated_data.csv")
            if os.path.isfile(data_file):
                df = pd.read_csv(data_file)
            else:
                split_art = run.use_artifact("splits:latest")
                split_path = split_art.download(root=tmp_dir)
                train_df = pd.read_csv(os.path.join(split_path, "train.csv"))
                valid_df = pd.read_csv(os.path.join(split_path, "valid.csv"))
                test_df = pd.read_csv(os.path.join(split_path, "test.csv"))
                df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

        if df.empty:
            logger.warning("Loaded dataframe is empty")

        wandb.summary["train_data_hash"] = df_hash(df)
        schema = {c: str(t) for c, t in df.dtypes.items()}
        wandb.summary["model_train_schema"] = schema

        schema_path = PROJECT_ROOT / "artifacts" / "model_schema.json"
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)
        sample_path = PROJECT_ROOT / "artifacts" / "train_sample_rows.csv"
        df.head(50).to_csv(sample_path, index=False)

        schema_art = wandb.Artifact(
            "model_schema", type="schema"
        )
        schema_art.add_file(str(schema_path))
        schema_art.add_file(str(sample_path))
        run.log_artifact(schema_art, aliases=["latest"])

        if cfg.data_load.get("log_sample_artifacts", True):
            wandb.log({"train_sample_rows": wandb.Table(dataframe=df.head(50))})

        run_model_pipeline(df, cfg_dict)

        if cfg.data_load.get("log_artifacts", True):
            art_specs = [
                ("model", "model.pkl", "model"),
                ("preprocessing_pipeline", "preprocessing_pipeline.pkl", "pipeline"),
                ("metrics_path", "metrics.json", "metrics"),
            ]
            for cfg_key, default_name, art_type in art_specs:
                p = PROJECT_ROOT / cfg.artifacts.get(cfg_key, f"models/{default_name}")
                if p.is_file():
                    art = wandb.Artifact(art_type, type=art_type)
                    if art_type == "model":
                        art.add_file(str(p))
                        pp_path = PROJECT_ROOT / cfg.artifacts.get(
                            "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
                        )
                        if pp_path.is_file():
                            art.add_file(str(pp_path))
                        art.add_file(str(schema_path))
                        art.add_file(str(sample_path))
                    else:
                        art.add_file(str(p))
                    run.log_artifact(art, aliases=["latest"])
                    logger.info("Logged %s artifact to W&B", art_type)

            splits_dir = PROJECT_ROOT / cfg.artifacts.get("splits_dir", "data/splits")
            split_files = [splits_dir / f for f in ["train.csv", "valid.csv", "test.csv"]]
            if all(f.is_file() for f in split_files):
                splits_art = wandb.Artifact("splits", type="dataset")
                for f in split_files:
                    splits_art.add_file(str(f))
                run.log_artifact(splits_art, aliases=["latest"])
                logger.info("Logged splits artifact to W&B")

            processed_dir = PROJECT_ROOT / cfg.artifacts.get("processed_dir", "data/processed")
            proc_files = [processed_dir / f for f in [
                "train_processed.csv", "valid_processed.csv", "test_processed.csv"
            ]]
            if all(p.is_file() for p in proc_files):
                proc_art = wandb.Artifact("processed_data", type="dataset")
                for p in proc_files:
                    proc_art.add_file(str(p))
                run.log_artifact(proc_art, aliases=["latest"])
                logger.info("Logged processed data artifact to W&B")

    except Exception as e:
        logger.exception("Model step failed")
        run.alert(title="Model Step Error", text=str(e))
        sys.exit(1)
    finally:
        wandb.finish()
        logger.info("WandB run finished")

if __name__ == "__main__":
    main()
