"""
data_validation/run.py

MLflow-compatible, modular data validation step with Hydra config, W&B logging, and robust error handling.
"""

import sys
import logging
import os
import hydra
import wandb
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import json
import yaml
import pandas as pd
import tempfile
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_validation.data_validation import validate_schema

load_dotenv() 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("data_validation")


def _html_from_report(report: Dict[str, Any]) -> str:
    """Create a simple HTML summary table from the validation report."""
    lines = ["<h2>Data Validation Report</h2>"]
    result = report.get("result", "unknown")
    lines.append(f"<p><b>Result:</b> {result}</p>")
    lines.append(
        f"<p>Errors: {len(report.get('errors', []))} | Warnings: {len(report.get('warnings', []))}</p>"
    )

    if report.get("errors"):
        lines.append("<h3>Errors</h3><ul>")
        for e in report["errors"]:
            lines.append(f"<li>{e}</li>")
        lines.append("</ul>")

    if report.get("warnings"):
        lines.append("<h3>Warnings</h3><ul>")
        for w in report["warnings"]:
            lines.append(f"<li>{w}</li>")
        lines.append("</ul>")

    details = report.get("details", {})
    if details:
        keys = set()
        for d in details.values():
            keys.update(d.keys())
        columns = ["column"] + sorted(keys)
        lines.append("<table border='1'><tr>")
        for c in columns:
            lines.append(f"<th>{c}</th>")
        lines.append("</tr>")
        for col, d in details.items():
            lines.append("<tr>")
            for c in columns:
                val = col if c == "column" else d.get(c, "")
                lines.append(f"<td>{val}</td>")
            lines.append("</tr>")
        lines.append("</table>")

    return "\n".join(lines)


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config_path = PROJECT_ROOT / "config.yaml"

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"data_validation_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="data_validation",
            name=run_name,
            config=dict(cfg),
            tags=["data_validation"]
        )
        logger.info("Started WandB run: %s", run_name)

        # Load raw data artifact from W&B
        raw_art = run.use_artifact("raw_data:latest")
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = raw_art.download(root=tmp_dir)
            df = pd.read_csv(os.path.join(raw_path, "raw_data.csv"))
        if df.empty:
            logger.warning("Loaded dataframe is empty.")

        # Load config dict for validation
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Resolve validation report path relative to project root so that
        # validate_data writes it to a predictable location regardless of
        # Hydra's working directory.
        val_report_rel = config_dict.get("data_validation", {}).get(
            "report_path", "logs/validation_report.json")
        val_report_full = PROJECT_ROOT / val_report_rel
        config_dict.setdefault("data_validation", {})[
            "report_path"] = str(val_report_full)

        validate_schema(df, config=config_dict)

        # Save validated data to a temporary CSV and log to W&B
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "validated_data.csv"
            df.to_csv(tmp_path, index=False)
            val_artifact = wandb.Artifact("validated_data", type="dataset")
            val_artifact.add_file(str(tmp_path), name="validated_data.csv")
            run.log_artifact(val_artifact, aliases=["latest"])
            logger.info("Logged validated data artifact to WandB")

        # Always log validation report to W&B (even if validation fails)
        val_report_full_path = val_report_full
        if val_report_full_path.is_file():
            artifact = wandb.Artifact("validation_report", type="report")
            artifact.add_file(str(val_report_full_path))
            run.log_artifact(artifact, aliases=["latest"])
            logger.info("Logged validation report to WandB")
            with open(val_report_full_path) as f:
                report = json.load(f)
            wandb.summary.update({
                "validation_result": report.get("result", "unknown"),
                "validation_errors": len(report.get("errors", [])),
                "validation_warnings": len(report.get("warnings", [])),
            })
            html = _html_from_report(report)
            wandb.log({"validation_report": wandb.Html(html)})
        else:
            logger.warning("Validation report not found for logging.")

    except Exception as e:
        logger.exception("Failed during data validation step")
        if run is not None:
            run.alert(title="Data Validation Error", text=str(e))
        # Always attempt to log the artifact if it exists
        if val_report_full.is_file() and wandb.run is not None:
            artifact = wandb.Artifact("validation_report", type="report")
            artifact.add_file(str(val_report_full))
            run.log_artifact(artifact, aliases=["latest"])
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()