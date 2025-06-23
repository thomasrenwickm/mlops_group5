import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

log = logging.getLogger("evaluation")

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluation")

    test_data_path = "/Users/massimot/Documents/IE/term_3/mlops/project/mlops_group5/data/splits/test.csv"
    model_path = "/Users/massimot/Documents/IE/term_3/mlops/project/mlops_group5/models/model.pkl"

    logger.info(f"Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    X_test = test_df.drop(columns=["SalePrice"])
    y_test = test_df["SalePrice"]

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info("Generating predictions")
    y_pred = model.predict(X_test)

    logger.info("Calculating evaluation metrics")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"RMSE: {rmse}, MAE: {mae}")

    if mlflow.active_run() is None:
        mlflow.start_run(run_name="evaluation")

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)

    # ✅ Save only the metrics
    Path("artifacts").mkdir(exist_ok=True)
    metrics_dict = {"rmse": rmse, "mae": mae}
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info("Saved evaluation metrics to artifacts/metrics.json")

if __name__ == "__main__":
    main()
