"""
model.py

Train and evaluate a regression model on the Ames Housing dataset.
Includes preprocessing, top-k feature selection,
metrics evaluation, and model persistence.
"""

import os
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from preprocess.preprocessing import (
    fit_and_save_pipeline,
)
from evaluation.evaluation import evaluate_regression

logger = logging.getLogger(__name__)


def run_model_pipeline(df_raw: pd.DataFrame, config: dict) -> None:
    try:
        logger.info("Starting model training pipeline...")

        # 1. Separate target
        target = config["target"]
        y_data = df_raw[target]
        x_data = df_raw.drop(columns=[target])

        # 2. Preprocessing
        x_processed = fit_and_save_pipeline(x_data, config)
        x_processed.columns = x_processed.columns.astype(str)

        # Align y
        y_data = y_data.loc[x_processed.index]

        # 3. Feature selection
        important_manual_features = config["features"]["most_relevant_features"]
        config_engineer_features = config["features"].get("engineered", [])
        all_selected_features = important_manual_features + config_engineer_features

        always_keep = [
            f for f in all_selected_features if f in x_processed.columns]

        selection_input = x_processed.drop(
            columns=always_keep, errors="ignore")
        selector = SelectKBest(score_func=f_regression, k=10)
        x_selected = selector.fit_transform(selection_input, y_data)
        selected_features = selection_input.columns[selector.get_support(
        )].tolist()
        selected_features.extend(always_keep)
        selected_features = list(set(selected_features))

        # Final data
        x_final = x_processed[selected_features]

        # 4. Train/Valid/Test split
        split_cfg = config["data_split"]
        test_size = split_cfg["test_size"]
        valid_size = split_cfg["valid_size"]
        random_state = split_cfg["random_state"]

        x_train, x_temp, y_train, y_temp = train_test_split(
            x_final, y_data, test_size=(test_size + valid_size), random_state=random_state
        )
        rel_valid = valid_size / (test_size + valid_size)
        x_valid, x_test, y_valid, y_test = train_test_split(
            x_temp, y_temp, test_size=rel_valid, random_state=random_state
        )

        # 5. Model training
        model_type = config["model"]["active"]
        if model_type == "linear_regression":
            model = LinearRegression()
        else:
            raise ValueError("Unsupported model type: %s", model_type)

        model.fit(x_train, y_train)
        logger.info("Model training completed.")

        # 6. Evaluation
        y_valid_pred = model.predict(x_valid)
        valid_metrics = evaluate_regression(y_valid, y_valid_pred)
        logger.info("Validation metrics: %s", valid_metrics)

        y_test_pred = model.predict(x_test)
        test_metrics = evaluate_regression(y_test, y_test_pred)
        logger.info("Test metrics: %s", test_metrics)

        # 7. Save metrics
        metrics_path = config["artifacts"]["metrics_path"]
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        pd.DataFrame({"validation": valid_metrics, "test": test_metrics}).to_json(
            metrics_path, indent=2
        )
        logger.info("Saved metrics to %s", metrics_path)

        # 8. Save model
        model_path = config["artifacts"]["model"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved model to %s", model_path)

        # 9. Save selected features
        features_path = config["artifacts"]["selected_features"]
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        pd.Series(selected_features).to_json(features_path)
        logger.info("Saved selected features to %s", features_path)

    except Exception as e:
        logger.exception("Model training pipeline failed: %s", e)
        raise
