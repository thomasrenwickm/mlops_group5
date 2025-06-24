import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.model.model import run_model_pipeline


class TestRunModelPipeline:

    @pytest.fixture
    def df_raw(self):
        return pd.DataFrame({
            'feature1': np.arange(10),
            'feature2': np.arange(0, 20, 2),
            'feature3': np.arange(5, 15),
            'important_feature': np.linspace(10, 100, 10),
            'SalePrice': np.linspace(100000, 200000, 10)
        })

    @pytest.fixture
    def config(self):
        return {
            "target": "SalePrice",
            "features": {
                "most_relevant_features": ["important_feature"],
                "engineered": ["feature1"]
            },
            "data_split": {
                "test_size": 0.2,
                "valid_size": 0.2,
                "random_state": 42
            },
            "model": {
                "active": "linear_regression"
            },
            "artifacts": {
                "metrics_path": "models/test_metrics.json",
                "model": "models/test_model.pkl",
                "selected_features": "models/test_features.json",
                "splits_dir": "tests/tmp/splits"
            },
            "preprocessing": {
                "rename_columns": {},
                "drop_columns": [],
                "fillna": {},
                "dropna_rows": []
            }
        }

    @patch("src.model.model.evaluate_regression")
    @patch("src.model.model.fit_and_save_pipeline")
    def test_pipeline_runs_and_saves(self, mock_fit_pipeline, mock_evaluate, df_raw, config):
        mock_fit_pipeline.return_value = df_raw.drop(columns=["SalePrice"])
        mock_evaluate.return_value = {
            "mse": 1000.0,
            "rmse": 31.6,
            "mae": 25.0,
            "r2": 0.95
        }

        run_model_pipeline(df_raw, config)

        # Validate mocks were used
        mock_fit_pipeline.assert_called_once()
        assert mock_evaluate.call_count == 2  # once for valid, once for test

    @patch("src.model.model.fit_and_save_pipeline")
    def test_unsupported_model_raises(self, mock_fit_pipeline, df_raw, config):
        config["model"]["active"] = "not_supported"
        mock_fit_pipeline.return_value = df_raw.drop(columns=["SalePrice"])

        with pytest.raises(ValueError, match="Unsupported model type: not_supported"):
            run_model_pipeline(df_raw, config)

    def test_config_structure(self, config):
        for key in ["target", "features", "data_split", "model", "artifacts"]:
            assert key in config

    def test_split_ratios(self, config):
        total = config["data_split"]["test_size"] + \
            config["data_split"]["valid_size"]
        assert 0 < total < 1
