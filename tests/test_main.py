import pytest
from unittest import mock
import sys

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src import main
# We'll import main.py as a module


@mock.patch("main.run_inference")
@mock.patch("main.run_model_pipeline")
@mock.patch("main.validate_schema")
@mock.patch("main.get_data")
@mock.patch("main.load_config")
def test_main_data_stage(
    mock_load_config,
    mock_get_data,
    mock_validate_schema,
    mock_run_model_pipeline,
    mock_run_inference,
):
    # Simulate CLI args for "data" stage
    test_args = [
        "main.py",
        "--stage", "data",
        "--config", "config.yaml"
    ]
    with mock.patch.object(sys, "argv", test_args):
        # Minimal dummy config
        dummy_config = {
            "data_validation": {
                "enabled": True,
                "schema": {
                    "columns": [
                        {"name": "Id", "dtype": "int", "required": True}
                    ]
                },
                "action_on_error": "raise"
            },
            "logging": {}
        }
        mock_load_config.return_value = dummy_config
        mock_get_data.return_value = mock.Mock(shape=(100, 10))

        # Run main
        main.main()

        # Assert calls
        mock_get_data.assert_called_once()
        mock_validate_schema.assert_called_once()
        mock_run_model_pipeline.assert_not_called()
        mock_run_inference.assert_not_called()

@mock.patch("main.run_inference")
@mock.patch("main.run_model_pipeline")
@mock.patch("main.validate_schema")
@mock.patch("main.get_data")
@mock.patch("main.load_config")
def test_main_train_stage(mock_load_config, mock_get_data, mock_validate_schema, mock_run_model_pipeline, mock_run_inference):
    test_args = [
        "main.py",
        "--stage", "train",
        "--config", "config.yaml"
    ]
    with mock.patch.object(sys, "argv", test_args):
        dummy_config = {
            "data_validation": {
                "enabled": True,
                "schema": {"columns": [{"name": "Id", "dtype": "int", "required": True}]},
                "action_on_error": "raise"
            },
            "target": "SalePrice",
            "logging": {}
        }
        mock_load_config.return_value = dummy_config
        mock_get_data.return_value = mock.Mock(shape=(100, 10))

        main.main()

        mock_run_model_pipeline.assert_called_once()
        mock_run_inference.assert_not_called()

@mock.patch("main.run_inference")
@mock.patch("main.run_model_pipeline")
@mock.patch("main.validate_schema")
@mock.patch("main.get_data")
@mock.patch("main.load_config")
def test_main_infer_stage_missing_args(mock_load_config, mock_get_data, mock_validate_schema, mock_run_model_pipeline, mock_run_inference):
    test_args = [
        "main.py",
        "--stage", "infer"
    ]
    with mock.patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit):  # Should exit due to missing args
            main.main()
