"""
model.py

Train and evaluate a regression model on the Ames Housing dataset.
Includes preprocessing, feature engineering, top-k feature selection,
metrics evaluation, and model persistence.
"""

import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.preprocess.preprocessing import preprocess_data
from src.features.features import engineer_features

logger = logging.getLogger(__name__)


def run_model_pipeline(df_raw: pd.DataFrame, config: dict) -> None:
    """
    Main function for training the model pipeline:
    - Feature engineer new variables
    - Preprocess data (via preprocess_data)
    - Select top-k features
    - Train model
    - Evaluate and save artifacts
    """
    try:
        logger.info("Starting model training pipeline...")

        # Extract target before preprocessing
        target = config["target"]
        y_data = df_raw[target]
        x_data = df_raw.drop(columns=[target])

        # Apply feature engineering from external module
        x_data = engineer_features(x_data, config)

        # Preprocess features only
        x_processed = preprocess_data(x_data)
        x_processed.columns = x_processed.columns.astype(str)

        important_manual_features = [
            "total_sf", "bathrooms", "house_age", "since_remodel",
            "overall_qual", "garage_cars", "gr_liv_area",
            "kitchen_qual", "exter_qual", "neighborhood"
        ]
        always_keep = [
            f for f in important_manual_features if f in x_processed.columns]

        # Drop them temporarily for selection
        selection_input = x_processed.drop(
            columns=always_keep, errors="ignore")

        # Select top k from remaining
        selector = SelectKBest(score_func=f_regression, k=10)
        x_selected = selector.fit_transform(selection_input, y_data)
        selected_features = selection_input.columns[selector.get_support(
        )].tolist()

        # Add important manual features
        selected_features.extend(always_keep)
        engineered_included = always_keep.copy()

        logger.info("Top features selected by SelectKBest: %s",
                    selected_features)

        x_final = x_processed[selected_features]

        # Split data
        split_cfg = config["data_split"]
        x_train, x_test, y_train, y_test = train_test_split(
            x_final,
            y_data,
            test_size=split_cfg["test_size"],
            random_state=split_cfg["random_state"]
        )

        # Train model
        model_type = config["model"]["active"]
        if model_type == "linear_regression":
            model = LinearRegression()
        else:
            raise ValueError("Unsupported model type: %s", model_type)

        model.fit(x_train, y_train)
        logger.info("Model training completed.")

        # Evaluate
        y_pred = model.predict(x_test)

        logger.info("Evaluation metrics: %s", metrics)

        # Save metrics
        metrics_path = config["artifacts"]["metrics_path"]
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        pd.Series(metrics).to_json(metrics_path, indent=2)
        logger.info("Saved metrics to %s", metrics_path)

        # Save model only (pipeline is saved in preprocess_data)
        model_path = config["model"].get("model_path", "models/model.pkl")
        joblib.dump(model, model_path)
        logger.info("Saved model to %s", model_path)

    except Exception as e:
        logger.exception("Model training pipeline failed: %s", e)
        raise
