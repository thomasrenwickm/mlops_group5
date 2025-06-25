# """
# test_main.py

# Simple unit tests for main.py module
# """
# import pytest
# import os
# import sys
# import logging
# import pandas as pd
# import yaml
# from unittest.mock import patch, MagicMock, call, mock_open
# from io import StringIO

# # Import the functions to test
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src_path = os.path.join(project_root, 'src')

# # Add both paths to handle imports correctly
# sys.path.insert(0, project_root)
# sys.path.insert(0, src_path)

# from src.main import setup_logging, main


# @pytest.fixture(scope="module", autouse=True)
# def setup_config_file():
#     """Create a temporary config file for import, then clean up."""
#     temp_config = {
#         'data_validation': {
#             'enabled': True,
#             'schema': {'columns': ['test']},
#             'action_on_error': 'raise'
#         }
#     }

#     config_path = 'config.yaml'
#     config_existed = os.path.exists(config_path)

#     # Create config file if it doesn't exist
#     if not config_existed:
#         with open(config_path, 'w') as f:
#             yaml.dump(temp_config, f)

#     yield

#     # Clean up only if we created it
#     if not config_existed and os.path.exists(config_path):
#         os.remove(config_path)


# @pytest.fixture
# def sample_config():
#     """Sample configuration for testing."""
#     return {
#         "logging": {
#             "level": "INFO",
#             "log_file": "logs/test.log",
#             "format": "%(asctime)s - %(levelname)s - %(message)s",
#             "datefmt": "%Y-%m-%d %H:%M:%S"
#         },
#         "data_validation": {
#             "enabled": True,
#             "schema": {
#                 "columns": ["feature1", "feature2"]
#             },
#             "action_on_error": "raise"
#         }
#     }


# @pytest.fixture
# def sample_dataframe():
#     """Sample DataFrame for testing."""
#     return pd.DataFrame({
#         'feature1': [1, 2, 3],
#         'feature2': [0.1, 0.2, 0.3],
#         'target': [100, 200, 300]
#     })


# class TestSetupLogging:
#     """Test cases for setup_logging function."""

#     @patch('os.makedirs')
#     @patch('logging.basicConfig')
#     @patch('logging.StreamHandler')
#     @patch('logging.getLogger')
#     def test_setup_logging_with_config(self, mock_get_logger, mock_stream_handler,
#                                        mock_basic_config, mock_makedirs):
#         """Test logging setup with full configuration."""
#         log_cfg = {
#             "level": "DEBUG",
#             "log_file": "logs/test.log",
#             "format": "%(asctime)s - %(name)s - %(message)s",
#             "datefmt": "%Y-%m-%d %H:%M:%S"
#         }

#         # Mock logger and handler
#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger
#         mock_handler = MagicMock()
#         mock_stream_handler.return_value = mock_handler

#         setup_logging(log_cfg)

#         # Verify directory creation
#         mock_makedirs.assert_called_once_with("logs", exist_ok=True)

#         # Verify basic config
#         mock_basic_config.assert_called_once_with(
#             level=logging.DEBUG,
#             format="%(asctime)s - %(name)s - %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S",
#             filename="logs/test.log",
#             filemode="a"
#         )

#         # Verify console handler setup
#         mock_stream_handler.assert_called_once()
#         mock_handler.setFormatter.assert_called_once()
#         mock_logger.addHandler.assert_called_once_with(mock_handler)

#     @patch('os.makedirs')
#     @patch('logging.basicConfig')
#     @patch('logging.StreamHandler')
#     @patch('logging.getLogger')
#     def test_setup_logging_with_defaults(self, mock_get_logger, mock_stream_handler,
#                                          mock_basic_config, mock_makedirs):
#         """Test logging setup with default values."""
#         log_cfg = {}

#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger
#         mock_handler = MagicMock()
#         mock_stream_handler.return_value = mock_handler

#         setup_logging(log_cfg)

#         # Verify defaults are used
#         mock_makedirs.assert_called_once_with("logs", exist_ok=True)
#         mock_basic_config.assert_called_once_with(
#             level=logging.INFO,
#             format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S",
#             filename="logs/main.log",
#             filemode="a"
#         )

#     @patch('os.makedirs')
#     @patch('logging.basicConfig')
#     @patch('logging.StreamHandler')
#     @patch('logging.getLogger')
#     def test_setup_logging_invalid_level(self, mock_get_logger, mock_stream_handler,
#                                          mock_basic_config, mock_makedirs):
#         """Test logging setup with invalid log level."""
#         log_cfg = {"level": "INVALID_LEVEL"}

#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger
#         mock_handler = MagicMock()
#         mock_stream_handler.return_value = mock_handler

#         setup_logging(log_cfg)

#         # Should default to INFO level for invalid level
#         mock_basic_config.assert_called_once()
#         args, kwargs = mock_basic_config.call_args
#         assert kwargs['level'] == logging.INFO


# class TestMain:
#     """Test cases for main function."""

#     @patch('src.main.logging.getLogger')
#     @patch('src.main.setup_logging')
#     @patch('src.main.load_config')
#     @patch('sys.argv', ['main.py', '--stage', 'data'])
#     def test_main_data_stage_only(self, mock_load_config, mock_setup_logging, mock_get_logger, sample_config):
#         """Test main function with data stage only."""
#         # Setup mocks
#         mock_load_config.return_value = sample_config
#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger

#         with patch('src.main.get_data') as mock_get_data, \
#              patch('src.main.validate_schema') as mock_validate_schema:

#             mock_get_data.return_value = pd.DataFrame({'feature1': [1, 2], 'feature2': [0.1, 0.2]})

#             main()

#             # Verify function calls
#             mock_load_config.assert_called_once_with("config.yaml")
#             mock_setup_logging.assert_called_once_with(sample_config["logging"])
#             mock_get_data.assert_called_once_with(
#                 config_path="config.yaml",
#                 env_path=".env",
#                 data_stage="raw"
#             )
#             mock_validate_schema.assert_called_once()

#     @patch('src.main.logging.getLogger')
#     @patch('src.main.setup_logging')
#     @patch('src.main.load_config')
#     @patch('sys.argv', ['main.py', '--stage', 'train'])
#     def test_main_train_stage_only(self, mock_load_config, mock_setup_logging, mock_get_logger, sample_config, sample_dataframe):
#         """Test main function with train stage only."""
#         mock_load_config.return_value = sample_config
#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger

#         with patch('src.main.get_data') as mock_get_data, \
#              patch('src.main.validate_schema') as mock_validate_schema, \
#              patch('src.main.run_model_pipeline') as mock_run_model:

#             mock_get_data.return_value = sample_dataframe

#             main()

#             # Verify train stage calls
#             mock_get_data.assert_called_once()
#             mock_validate_schema.assert_called_once()
#             mock_run_model.assert_called_once_with(sample_dataframe, sample_config)

#     @patch('src.main.logging.getLogger')
#     @patch('src.main.setup_logging')
#     @patch('src.main.load_config')
#     @patch('sys.argv', ['main.py', '--stage', 'infer', '--input_csv', 'input.csv', '--output_csv', 'output.csv'])
#     def test_main_inference_stage(self, mock_load_config, mock_setup_logging, mock_get_logger, sample_config):
#         """Test main function with inference stage."""
#         mock_load_config.return_value = sample_config
#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger

#         with patch('src.main.run_inference') as mock_run_inference:
#             main()

#             # Verify inference stage calls
#             mock_run_inference.assert_called_once_with(
#                 input_csv="input.csv",
#                 config_path="config.yaml",
#                 output_csv="output.csv"
#             )

#     @patch('src.main.logging.getLogger')
#     @patch('src.main.setup_logging')
#     @patch('src.main.load_config')
#     @patch('sys.argv', ['main.py', '--stage', 'all'])
#     def test_main_all_stages(self, mock_load_config, mock_setup_logging, mock_get_logger, sample_config, sample_dataframe):
#         """Test main function with all stages."""
#         mock_load_config.return_value = sample_config
#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger

#         with patch('src.main.get_data') as mock_get_data, \
#              patch('src.main.validate_schema') as mock_validate_schema, \
#              patch('src.main.run_model_pipeline') as mock_run_model:

#             mock_get_data.return_value = sample_dataframe

#             main()

#             # Verify all stages are called
#             mock_get_data.assert_called_once()
#             mock_validate_schema.assert_called_once()
#             mock_run_model.assert_called_once_with(sample_dataframe, sample_config)

#     @patch('src.main.load_config')
#     @patch('sys.argv', ['main.py', '--config', 'missing.yaml'])
#     def test_main_config_not_found(self, mock_load_config):
#         """Test main function when config file is not found."""
#         mock_load_config.side_effect = FileNotFoundError("Config file not found")

#         with pytest.raises(SystemExit) as exc_info:
#             with patch('sys.stderr', new_callable=StringIO):
#                 main()

#         assert exc_info.value.code == 1

#     @patch('src.main.logging.getLogger')
#     @patch('src.main.setup_logging')
#     @patch('src.main.load_config')
#     @patch('sys.argv', ['main.py', '--stage', 'infer'])
#     def test_main_inference_missing_args(self, mock_load_config, mock_setup_logging, mock_get_logger, sample_config):
#         """Test main function inference stage without required CSV arguments."""
#         mock_load_config.return_value = sample_config
#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger

#         with pytest.raises(SystemExit) as exc_info:
#             main()

#         assert exc_info.value.code == 1
#         mock_logger.error.assert_called_once()

#     @patch('src.main.logging.getLogger')
#     @patch('src.main.setup_logging')
#     @patch('src.main.load_config')
#     @patch('sys.argv', ['main.py', '--stage', 'data'])
#     def test_main_pipeline_exception(self, mock_load_config, mock_setup_logging, mock_get_logger, sample_config):
#         """Test main function handling pipeline exceptions."""
#         mock_load_config.return_value = sample_config
#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger

#         with patch('src.main.get_data') as mock_get_data:
#             mock_get_data.side_effect = Exception("Pipeline error")

#             with pytest.raises(SystemExit) as exc_info:
#                 main()

#             assert exc_info.value.code == 1
#             mock_logger.exception.assert_called_once()

#     @patch('src.main.logging.getLogger')
#     @patch('src.main.setup_logging')
#     @patch('src.main.load_config')
#     @patch('sys.argv', ['main.py', '--config', 'custom.yaml', '--env', 'custom.env', '--stage', 'data'])
#     def test_main_custom_config_env(self, mock_load_config, mock_setup_logging, mock_get_logger, sample_config):
#         """Test main function with custom config and env paths."""
#         mock_load_config.return_value = sample_config
#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger

#         with patch('src.main.get_data') as mock_get_data, \
#              patch('src.main.validate_schema'):

#             mock_get_data.return_value = pd.DataFrame({'feature1': [1], 'feature2': [0.1]})

#             main()

#             # Verify custom paths are used
#             mock_load_config.assert_called_once_with("custom.yaml")
#             mock_get_data.assert_called_once_with(
#                 config_path="custom.yaml",
#                 env_path="custom.env",
#                 data_stage="raw"
#             )

#     @patch('src.main.logging.getLogger')
#     @patch('src.main.setup_logging')
#     @patch('src.main.load_config')
#     @patch('sys.argv', ['main.py', '--stage', 'data'])
#     def test_main_validation_disabled(self, mock_load_config, mock_setup_logging, mock_get_logger):
#         """Test main function with data validation disabled."""
#         config_no_validation = {
#             "logging": {},
#             "data_validation": {"enabled": False}
#         }
#         mock_load_config.return_value = config_no_validation
#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger

#         with patch('src.main.get_data') as mock_get_data, \
#              patch('src.main.validate_schema') as mock_validate_schema:

#             mock_get_data.return_value = pd.DataFrame({'feature1': [1], 'feature2': [0.1]})

#             main()

#             # Verify validation is not called
#             mock_get_data.assert_called_once()
#             mock_validate_schema.assert_not_called()


# class TestArgumentParsing:
#     """Test cases for argument parsing functionality."""

#     @patch('src.main.logging.getLogger')
#     @patch('src.main.setup_logging')
#     @patch('src.main.load_config')
#     def test_default_arguments(self, mock_load_config, mock_setup_logging, mock_get_logger, sample_config, sample_dataframe):
#         """Test main function with default arguments."""
#         mock_load_config.return_value = sample_config
#         mock_logger = MagicMock()
#         mock_get_logger.return_value = mock_logger

#         with patch('sys.argv', ['main.py']), \
#              patch('src.main.get_data') as mock_get_data, \
#              patch('src.main.validate_schema'), \
#              patch('src.main.run_model_pipeline'):

#             mock_get_data.return_value = sample_dataframe

#             main()

#             # Verify defaults are used
#             mock_load_config.assert_called_once_with("config.yaml")
#             mock_get_data.assert_called_with(
#                 config_path="config.yaml",
#                 env_path=".env",
#                 data_stage="raw"
#             )


# class TestIntegration:
#     """Integration tests for the main module."""

#     @patch('src.main.run_model_pipeline')
#     @patch('src.main.validate_schema')
#     @patch('src.main.get_data')
#     @patch('src.main.setup_logging')
#     @patch('src.main.load_config')
#     @patch('sys.argv', ['main.py', '--stage', 'all'])
#     def test_full_pipeline_workflow(self, mock_load_config, mock_setup_logging,
#                                     mock_get_data, mock_validate_schema,
#                                     mock_run_model, sample_config, sample_dataframe):
#         """Test complete pipeline workflow."""
#         # Setup mocks
#         mock_load_config.return_value = sample_config
#         mock_get_data.return_value = sample_dataframe

#         with patch('src.main.logging.getLogger') as mock_get_logger:
#             mock_logger = MagicMock()
#             mock_get_logger.return_value = mock_logger

#             main()

#             # Verify complete workflow
#             mock_load_config.assert_called_once()
#             mock_setup_logging.assert_called_once()
#             mock_get_data.assert_called_once()
#             mock_validate_schema.assert_called_once()
#             mock_run_model.assert_called_once()

#             # Verify logging calls
#             assert mock_logger.info.call_count >= 2  # Pipeline start and completion


# if __name__ == "__main__":
#     pytest.main([__file__])

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_minimal_payload():
    response = client.post("/predict", json={
        "PID": 526350040,
        "MS SubClass": 20,
        "MS Zoning": "RH",
        "Lot Frontage": 80.0,
        "Lot Area": 11622,
        "Street": "Pave",
        "Alley": 'NA',
        "Lot Shape": "Reg",
        "Land Contour": "Lvl",
        "Utilities": "AllPub",
        "Lot Config": "Inside",
        "Land Slope": "Gtl",
        "Neighborhood": "NAmes",
        "Condition 1": "Feedr",
        "Condition 2": "Norm",
        "Bldg Type": "1Fam",
        "House Style": "1Story",
        "Overall Qual": 5,
        "Overall Cond": 6,
        "Year Built": 1961,
        "Year Remod/Add": 1961,
        "Roof Style": "Gable",
        "Roof Matl": "CompShg",
        "Exterior 1st": "VinylSd",
        "Exterior 2nd": "VinylSd",
        "Mas Vnr Type": "None",
        "Mas Vnr Area": 0.0,
        "Exter Qual": "TA",
        "Exter Cond": "TA",
        "Foundation": "CBlock",
        "Bsmt Qual": "TA",
        "Bsmt Cond": "TA",
        "Bsmt Exposure": "No",
        "BsmtFin Type 1": "Rec",
        "BsmtFin SF 1": 468.0,
        "BsmtFin Type 2": "LwQ",
        "BsmtFin SF 2": 144.0,
        "Bsmt Unf SF": 270.0,
        "Total Bsmt SF": 882.0,
        "Heating": "GasA",
        "Heating QC": "TA",
        "Central Air": "Y",
        "Electrical": "SBrkr",
        "1st Flr SF": 896,
        "2nd Flr SF": 0,
        "Low Qual Fin SF": 0,
        "Gr Liv Area": 896,
        "Bsmt Full Bath": 0.0,
        "Bsmt Half Bath": 0.0,
        "Full Bath": 1,
        "Half Bath": 0,
        "Bedroom AbvGr": 2,
        "Kitchen AbvGr": 1,
        "Kitchen Qual": "TA",
        "TotRms AbvGr": 5,
        "Functional": "Typ",
        "Fireplaces": 0,
        "Fireplace Qu": 'NA',
        "Garage Type": "Attchd",
        "Garage Yr Blt": 1961.0,
        "Garage Finish": "Unf",
        "Garage Cars": 1.0,
        "Garage Area": 730.0,
        "Garage Qual": "TA",
        "Garage Cond": "TA",
        "Paved Drive": "Y",
        "Wood Deck SF": 140,
        "Open Porch SF": 0,
        "Enclosed Porch": 0,
        "3Ssn Porch": 0,
        "Screen Porch": 120,
        "Pool Area": 0,
        "Pool QC": 'NA',
        "Fence": "MnPrv",
        "Misc Feature": 'NA',
        "Misc Val": 0,
        "Mo Sold": 6,
        "Yr Sold": 2010,
        "Sale Type": "WD",
        "Sale Condition": "Normal"
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
