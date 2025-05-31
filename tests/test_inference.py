import pandas as pd
import pytest
import numpy as np
import sys
import os
import tempfile
import pickle
import json
import types
from unittest.mock import patch, mock_open, MagicMock, call
from pathlib import Path

# Add the parent directory to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Manually mock the dependencies before importing inference
# Create mock modules
preprocess_mock = types.ModuleType('preprocess')
preprocessing_mock = types.ModuleType('preprocess.preprocessing')
features_mock = types.ModuleType('features')
features_features_mock = types.ModuleType('features.features')

# Create mock functions
mock_transform_func = MagicMock()
mock_engineer_func = MagicMock()

# Add functions to mock modules
preprocessing_mock.transform_with_pipeline = mock_transform_func
features_features_mock.engineer_features = mock_engineer_func

# Add to sys.modules
sys.modules['preprocess'] = preprocess_mock
sys.modules['preprocess.preprocessing'] = preprocessing_mock
sys.modules['features'] = features_mock
sys.modules['features.features'] = features_features_mock

try:
    # Try different possible import paths based on your project structure
    from src.inference.inference import run_inference
    MODULE_PATH = 'src.inference.inference'
except ImportError:
    try:
        from src.inference import run_inference
        MODULE_PATH = 'src.inference'
    except ImportError:
        try:
            from inference import run_inference
            MODULE_PATH = 'inference'
        except ImportError:
            try:
                # If inference.py is directly in src/
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
                from inference import run_inference
                MODULE_PATH = 'inference'
            except ImportError:
                try:
                    # If inference.py is in src/inference/ directory
                    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'inference'))
                    from inference import run_inference
                    MODULE_PATH = 'inference'
                except ImportError:
                    print("Debug: Available files in src directory:")
                    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
                    if os.path.exists(src_path):
                        print(os.listdir(src_path))
                        
                        # Check inference subdirectory
                        inference_path = os.path.join(src_path, 'inference')
                        if os.path.exists(inference_path):
                            print("Files in src/inference directory:")
                            print(os.listdir(inference_path))
                    
                    raise ImportError("Could not import run_inference. Please check the module location.")

print(f"Successfully imported run_inference from {MODULE_PATH}")


# Simple class that can be pickled for integration tests
class MockModel:
    def predict(self, X):
        return np.array([100, 200, 300])

class MockPipeline:
    def transform(self, X):
        return X


class TestRunInference:
    """Test cases for run_inference function"""
    
    @patch(f'{MODULE_PATH}.logger')
    @patch(f'{MODULE_PATH}.engineer_features')
    @patch(f'{MODULE_PATH}.transform_with_pipeline')
    def test_run_inference_success_full_pipeline(self, mock_transform, mock_engineer, mock_logger):
        """Test successful inference run with full pipeline"""
        
        # Mock config data
        mock_config = {
            "artifacts": {
                "model": "models/model.pkl",
                "preprocessing_pipeline": "models/pipeline.pkl",
                "selected_features": "models/features.json"
            },
            "data_source": {},
            "feature_engineering": {"enabled": True}
        }
        
        # Mock input data
        input_data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "feature3": [7, 8, 9]
        })
        
        # Mock engineered data
        engineered_data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "feature3": [7, 8, 9],
            "engineered_feature": [10, 11, 12]
        })
        
        # Mock processed data
        processed_data = pd.DataFrame({
            "feature1_scaled": [0.1, 0.2, 0.3],
            "feature2_scaled": [0.4, 0.5, 0.6],
            "engineered_feature_scaled": [0.7, 0.8, 0.9]
        })
        
        # Mock selected features
        selected_features = ["feature1_scaled", "feature2_scaled"]
        
        # Mock model predictions
        predictions = np.array([100, 200, 300])
        
        # Mock objects
        mock_pipeline = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = predictions
        
        # Setup mocks
        mock_engineer.return_value = engineered_data
        mock_transform.return_value = processed_data
        
        # Create a mock that handles both text and binary modes
        def mock_open_handler(*args, **kwargs):
            if len(args) > 1 and 'b' in args[1]:
                # Binary mode for pickle files
                return mock_open(read_data=b'')()
            else:
                # Text mode for other files
                return mock_open(read_data='')()
        
        with patch('builtins.open', side_effect=mock_open_handler):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=mock_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', return_value=input_data):
                    with patch(f'{MODULE_PATH}.pickle.load', side_effect=[mock_pipeline, mock_model]):
                        with patch(f'{MODULE_PATH}.pd.read_json') as mock_read_json:
                            with patch('pathlib.Path') as mock_path:
                                with patch(f'{MODULE_PATH}.pd.DataFrame') as mock_df_constructor:
                                    
                                    # Setup remaining mocks
                                    mock_read_json.return_value.tolist.return_value = selected_features
                                    mock_path.return_value.parent.mkdir = MagicMock()
                                    
                                    # Mock DataFrame constructor and to_csv
                                    mock_result_df = MagicMock()
                                    mock_df_constructor.return_value = mock_result_df
                                    
                                    # Run inference
                                    run_inference("input.csv", "config.yaml", "output.csv")
        
        # Verify function calls
        mock_engineer.assert_called_once_with(input_data, mock_config)
        mock_transform.assert_called_once_with(engineered_data, mock_config, mock_pipeline)
        mock_model.predict.assert_called_once()
        
        # Verify logging
        assert mock_logger.info.call_count >= 7  # Should have multiple log messages
        
        # Verify file operations
        mock_df_constructor.assert_called_once_with({"prediction": predictions})
        mock_result_df.to_csv.assert_called_once_with("output.csv", index=False)
    
    def test_run_inference_config_file_not_found(self):
        """Test behavior when config file doesn't exist"""
        with patch('builtins.open', side_effect=FileNotFoundError("Config file not found")):
            with pytest.raises(FileNotFoundError):
                run_inference("input.csv", "nonexistent_config.yaml", "output.csv")
    
    def test_run_inference_input_csv_not_found(self):
        """Test behavior when input CSV doesn't exist"""
        mock_config = {
            "artifacts": {
                "model": "models/model.pkl",
                "preprocessing_pipeline": "models/pipeline.pkl", 
                "selected_features": "models/features.json"
            },
            "data_source": {}
        }
        
        # Create a mock that handles both text and binary modes
        def mock_open_handler(*args, **kwargs):
            if len(args) > 1 and 'b' in args[1]:
                return mock_open(read_data=b'')()
            else:
                return mock_open(read_data='')()
        
        with patch('builtins.open', side_effect=mock_open_handler):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=mock_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', side_effect=FileNotFoundError("Input CSV not found")):
                    with pytest.raises(FileNotFoundError):
                        run_inference("nonexistent.csv", "config.yaml", "output.csv")
    
    @patch(f'{MODULE_PATH}.logger')
    def test_run_inference_model_file_not_found(self, mock_logger):
        """Test behavior when model file doesn't exist"""
        mock_config = {
            "artifacts": {
                "model": "nonexistent_model.pkl",
                "preprocessing_pipeline": "models/pipeline.pkl",
                "selected_features": "models/features.json"
            },
            "data_source": {}
        }
        
        input_data = pd.DataFrame({"feature1": [1, 2, 3]})
        mock_pipeline = MagicMock()
        
        # Create a mock that handles both text and binary modes
        def mock_open_handler(*args, **kwargs):
            if len(args) > 1 and 'b' in args[1]:
                return mock_open(read_data=b'')()
            else:
                return mock_open(read_data='')()
        
        with patch('builtins.open', side_effect=mock_open_handler):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=mock_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', return_value=input_data):
                    with patch(f'{MODULE_PATH}.engineer_features', return_value=input_data):
                        with patch(f'{MODULE_PATH}.pickle.load', side_effect=[mock_pipeline, FileNotFoundError("Model not found")]):
                            with patch(f'{MODULE_PATH}.transform_with_pipeline', return_value=input_data):
                                with patch(f'{MODULE_PATH}.pd.read_json') as mock_read_json:
                                    mock_read_json.return_value.tolist.return_value = ["feature1"]
                                    
                                    with pytest.raises(FileNotFoundError):
                                        run_inference("input.csv", "config.yaml", "output.csv")
    
    @patch(f'{MODULE_PATH}.logger')
    @patch(f'{MODULE_PATH}.engineer_features')
    @patch(f'{MODULE_PATH}.transform_with_pipeline')
    def test_run_inference_config_structure_validation(self, mock_transform, mock_engineer, mock_logger):
        """Test behavior with missing required config keys"""
        
        # Config missing required keys
        incomplete_config = {
            "artifacts": {
                "model": "models/model.pkl"
                # Missing pipeline and features paths
            },
            "data_source": {}
        }
        
        input_data = pd.DataFrame({"feature1": [1, 2, 3]})
        
        # Create a mock that handles both text and binary modes
        def mock_open_handler(*args, **kwargs):
            if len(args) > 1 and 'b' in args[1]:
                return mock_open(read_data=b'')()
            else:
                return mock_open(read_data='')()
        
        with patch('builtins.open', side_effect=mock_open_handler):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=incomplete_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', return_value=input_data):
                    with pytest.raises(KeyError):
                        run_inference("input.csv", "config.yaml", "output.csv")
    
    @patch(f'{MODULE_PATH}.logger')
    @patch(f'{MODULE_PATH}.engineer_features')
    def test_run_inference_feature_engineering_error(self, mock_engineer, mock_logger):
        """Test behavior when feature engineering fails"""
        mock_config = {
            "artifacts": {
                "model": "models/model.pkl",
                "preprocessing_pipeline": "models/pipeline.pkl",
                "selected_features": "models/features.json"
            },
            "data_source": {}
        }
        
        input_data = pd.DataFrame({"feature1": [1, 2, 3]})
        
        # Make feature engineering fail
        mock_engineer.side_effect = ValueError("Feature engineering failed")
        
        # Create a mock that handles both text and binary modes
        def mock_open_handler(*args, **kwargs):
            if len(args) > 1 and 'b' in args[1]:
                return mock_open(read_data=b'')()
            else:
                return mock_open(read_data='')()
        
        with patch('builtins.open', side_effect=mock_open_handler):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=mock_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', return_value=input_data):
                    with pytest.raises(ValueError):
                        run_inference("input.csv", "config.yaml", "output.csv")
    
    @patch(f'{MODULE_PATH}.logger')
    @patch(f'{MODULE_PATH}.engineer_features')
    @patch(f'{MODULE_PATH}.transform_with_pipeline')
    def test_run_inference_preprocessing_error(self, mock_transform, mock_engineer, mock_logger):
        """Test behavior when preprocessing fails"""
        mock_config = {
            "artifacts": {
                "model": "models/model.pkl",
                "preprocessing_pipeline": "models/pipeline.pkl",
                "selected_features": "models/features.json"
            },
            "data_source": {}
        }
        
        input_data = pd.DataFrame({"feature1": [1, 2, 3]})
        mock_pipeline = MagicMock()
        
        mock_engineer.return_value = input_data
        mock_transform.side_effect = ValueError("Preprocessing failed")
        
        # Create a mock that handles both text and binary modes
        def mock_open_handler(*args, **kwargs):
            if len(args) > 1 and 'b' in args[1]:
                return mock_open(read_data=b'')()
            else:
                return mock_open(read_data='')()
        
        with patch('builtins.open', side_effect=mock_open_handler):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=mock_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', return_value=input_data):
                    with patch(f'{MODULE_PATH}.pickle.load', return_value=mock_pipeline):
                        with patch(f'{MODULE_PATH}.pd.read_json') as mock_read_json:
                            mock_read_json.return_value.tolist.return_value = ["feature1"]
                            with pytest.raises(ValueError):
                                run_inference("input.csv", "config.yaml", "output.csv")
    
    @patch(f'{MODULE_PATH}.logger')
    @patch(f'{MODULE_PATH}.engineer_features')
    @patch(f'{MODULE_PATH}.transform_with_pipeline')
    def test_run_inference_model_prediction_error(self, mock_transform, mock_engineer, mock_logger):
        """Test behavior when model prediction fails"""
        mock_config = {
            "artifacts": {
                "model": "models/model.pkl",
                "preprocessing_pipeline": "models/pipeline.pkl",
                "selected_features": "models/features.json"
            },
            "data_source": {}
        }
        
        input_data = pd.DataFrame({"feature1": [1, 2, 3]})
        processed_data = pd.DataFrame({"feature1_scaled": [0.1, 0.2, 0.3]})
        
        mock_pipeline = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.side_effect = ValueError("Model prediction failed")
        
        mock_engineer.return_value = input_data
        mock_transform.return_value = processed_data
        
        with patch('builtins.open', mock_open()):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=mock_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', return_value=input_data):
                    with patch(f'{MODULE_PATH}.pickle.load', side_effect=[mock_pipeline, mock_model]):
                        with patch(f'{MODULE_PATH}.pd.read_json') as mock_read_json:
                            mock_read_json.return_value.tolist.return_value = ["feature1_scaled"]
                            
                            with pytest.raises(ValueError):
                                run_inference("input.csv", "config.yaml", "output.csv")
    
    @patch(f'{MODULE_PATH}.logger')
    @patch(f'{MODULE_PATH}.engineer_features')
    @patch(f'{MODULE_PATH}.transform_with_pipeline')
    def test_run_inference_output_directory_creation(self, mock_transform, mock_engineer, mock_logger):
        """Test that output directory is created if it doesn't exist"""
        mock_config = {
            "artifacts": {
                "model": "models/model.pkl",
                "preprocessing_pipeline": "models/pipeline.pkl",
                "selected_features": "models/features.json"
            },
            "data_source": {}
        }
        
        input_data = pd.DataFrame({"feature1": [1, 2, 3]})
        processed_data = pd.DataFrame({"feature1_scaled": [0.1, 0.2, 0.3]})
        predictions = np.array([100, 200, 300])
        
        mock_pipeline = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = predictions
        
        mock_engineer.return_value = input_data
        mock_transform.return_value = processed_data
        
        with patch('builtins.open', mock_open()):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=mock_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', return_value=input_data):
                    with patch(f'{MODULE_PATH}.pickle.load', side_effect=[mock_pipeline, mock_model]):
                        with patch(f'{MODULE_PATH}.pd.read_json') as mock_read_json:
                            with patch('pathlib.Path') as mock_path:
                                with patch(f'{MODULE_PATH}.pd.DataFrame') as mock_df_constructor:
                                    
                                    mock_read_json.return_value.tolist.return_value = ["feature1_scaled"]
                                    mock_result_df = MagicMock()
                                    mock_df_constructor.return_value = mock_result_df
                                    
                                    # Mock Path behavior
                                    mock_path_instance = MagicMock()
                                    mock_path.return_value = mock_path_instance
                                    
                                    run_inference("input.csv", "config.yaml", "output/predictions.csv")
                                    
                                    # Verify directory creation was called
                                    mock_path.assert_called_once_with("output/predictions.csv")
                                    mock_path_instance.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @patch(f'{MODULE_PATH}.logger')
    @patch(f'{MODULE_PATH}.engineer_features')
    @patch(f'{MODULE_PATH}.transform_with_pipeline')
    def test_run_inference_config_updates(self, mock_transform, mock_engineer, mock_logger):
        """Test that config is properly updated with input/output paths"""
        original_config = {
            "artifacts": {
                "model": "models/model.pkl",
                "preprocessing_pipeline": "models/pipeline.pkl",
                "selected_features": "models/features.json"
            },
            "data_source": {"existing_key": "existing_value"}
        }
        
        input_data = pd.DataFrame({"feature1": [1, 2, 3]})
        processed_data = pd.DataFrame({"feature1_scaled": [0.1, 0.2, 0.3]})
        predictions = np.array([100, 200, 300])
        
        mock_pipeline = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = predictions
        
        mock_engineer.return_value = input_data
        mock_transform.return_value = processed_data
        
        with patch('builtins.open', mock_open()):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=original_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', return_value=input_data):
                    with patch(f'{MODULE_PATH}.pickle.load', side_effect=[mock_pipeline, mock_model]):
                        with patch(f'{MODULE_PATH}.pd.read_json') as mock_read_json:
                            with patch('pathlib.Path'):
                                with patch(f'{MODULE_PATH}.pd.DataFrame') as mock_df_constructor:
                                    
                                    mock_read_json.return_value.tolist.return_value = ["feature1_scaled"]
                                    mock_result_df = MagicMock()
                                    mock_df_constructor.return_value = mock_result_df
                                    
                                    run_inference("test_input.csv", "config.yaml", "test_output.csv")
                                    
                                    # Verify config was passed to functions with updates
                                    called_config = mock_engineer.call_args[0][1]
                                    assert called_config["data_source"]["new_data_path"] == "test_input.csv"
                                    assert called_config["artifacts"]["inference_output"] == "test_output.csv"
                                    assert called_config["data_source"]["existing_key"] == "existing_value"  # Preserved
    
    @patch(f'{MODULE_PATH}.logger')
    @patch(f'{MODULE_PATH}.engineer_features')
    @patch(f'{MODULE_PATH}.transform_with_pipeline')
    def test_run_inference_feature_selection(self, mock_transform, mock_engineer, mock_logger):
        """Test that only selected features are used for prediction"""
        mock_config = {
            "artifacts": {
                "model": "models/model.pkl",
                "preprocessing_pipeline": "models/pipeline.pkl",
                "selected_features": "models/features.json"
            },
            "data_source": {}
        }
        
        input_data = pd.DataFrame({"feature1": [1, 2, 3]})
        
        # Processed data has more features than selected
        processed_data = pd.DataFrame({
            "feature1_scaled": [0.1, 0.2, 0.3],
            "feature2_scaled": [0.4, 0.5, 0.6],
            "feature3_scaled": [0.7, 0.8, 0.9]
        })
        
        # Only some features are selected
        selected_features = ["feature1_scaled", "feature3_scaled"]
        predictions = np.array([100, 200, 300])
        
        mock_pipeline = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = predictions
        
        mock_engineer.return_value = input_data
        mock_transform.return_value = processed_data
        
        with patch('builtins.open', mock_open()):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=mock_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', return_value=input_data):
                    with patch(f'{MODULE_PATH}.pickle.load', side_effect=[mock_pipeline, mock_model]):
                        with patch(f'{MODULE_PATH}.pd.read_json') as mock_read_json:
                            with patch('pathlib.Path'):
                                with patch(f'{MODULE_PATH}.pd.DataFrame') as mock_df_constructor:
                                    
                                    mock_read_json.return_value.tolist.return_value = selected_features
                                    mock_result_df = MagicMock()
                                    mock_df_constructor.return_value = mock_result_df
                                    
                                    run_inference("input.csv", "config.yaml", "output.csv")
                                    
                                    # Verify model was called with only selected features
                                    actual_call_args = mock_model.predict.call_args[0][0]
                                    expected_features = processed_data[selected_features]
                                    
                                    # Check that the right columns were selected
                                    assert list(actual_call_args.columns) == selected_features
    
    @patch(f'{MODULE_PATH}.logger')
    def test_run_inference_empty_input_data(self, mock_logger):
        """Test behavior with empty input data"""
        mock_config = {
            "artifacts": {
                "model": "models/model.pkl",
                "preprocessing_pipeline": "models/pipeline.pkl",
                "selected_features": "models/features.json"
            },
            "data_source": {}
        }
        
        # Empty DataFrame
        empty_data = pd.DataFrame()
        
        # Create a mock that handles both text and binary modes
        def mock_open_handler(*args, **kwargs):
            if len(args) > 1 and 'b' in args[1]:
                return mock_open(read_data=b'')()
            else:
                return mock_open(read_data='')()
        
        with patch('builtins.open', side_effect=mock_open_handler):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=mock_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', return_value=empty_data):
                    with patch(f'{MODULE_PATH}.engineer_features', return_value=empty_data):
                        with patch(f'{MODULE_PATH}.pickle.load') as mock_pickle_load:
                            with patch(f'{MODULE_PATH}.pd.read_json') as mock_read_json:
                                with patch(f'{MODULE_PATH}.transform_with_pipeline', return_value=empty_data):
                                    # Setup mocks
                                    mock_pipeline = MagicMock()
                                    mock_model = MagicMock()
                                    mock_pickle_load.side_effect = [mock_pipeline, mock_model]
                                    mock_read_json.return_value.tolist.return_value = ["feature1"]
                                    
                                    # Should handle empty data gracefully or raise appropriate error
                                    # This depends on your specific implementation
                                    with pytest.raises((ValueError, KeyError, IndexError)):
                                        run_inference("empty.csv", "config.yaml", "output.csv")


class TestRunInferenceIntegration:
    """Integration tests for run_inference function"""
    
    def test_run_inference_with_real_files(self):
        """Test inference with actual file operations (using temporary files)"""
        
        # Create temporary config
        config_data = {
            "artifacts": {
                "model": "temp_model.pkl",
                "preprocessing_pipeline": "temp_pipeline.pkl",
                "selected_features": "temp_features.json"
            },
            "data_source": {}
        }
        
        # Create sample data
        input_data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6]
        })
        
        # Create real model and pipeline objects that can be pickled
        mock_model = MockModel()
        mock_pipeline = MockPipeline()
        
        selected_features = ["feature1", "feature2"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup file paths
            config_path = os.path.join(temp_dir, "config.yaml")
            input_path = os.path.join(temp_dir, "input.csv")
            output_path = os.path.join(temp_dir, "output.csv")
            model_path = os.path.join(temp_dir, "temp_model.pkl")
            pipeline_path = os.path.join(temp_dir, "temp_pipeline.pkl")
            features_path = os.path.join(temp_dir, "temp_features.json")
            
            # Update config with correct paths
            config_data["artifacts"]["model"] = model_path
            config_data["artifacts"]["preprocessing_pipeline"] = pipeline_path
            config_data["artifacts"]["selected_features"] = features_path
            
            # Write files
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            input_data.to_csv(input_path, index=False)
            
            with open(model_path, 'wb') as f:
                pickle.dump(mock_model, f)
            
            with open(pipeline_path, 'wb') as f:
                pickle.dump(mock_pipeline, f)
            
            with open(features_path, 'w') as f:
                json.dump(selected_features, f)
            
            # Mock the functions that we can't easily create files for
            with patch(f'{MODULE_PATH}.engineer_features', return_value=input_data):
                with patch(f'{MODULE_PATH}.transform_with_pipeline', return_value=input_data):
                    with patch(f'{MODULE_PATH}.logger'):
                        # Run inference
                        run_inference(input_path, config_path, output_path)
                        
                        # Verify output file was created
                        assert os.path.exists(output_path)
                        
                        # Verify output content
                        result_df = pd.read_csv(output_path)
                        assert "prediction" in result_df.columns
                        assert len(result_df) == 3
                        assert result_df["prediction"].tolist() == [100, 200, 300]


# Fixtures for common test data
@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "artifacts": {
            "model": "models/trained_model.pkl",
            "preprocessing_pipeline": "models/preprocessing_pipeline.pkl",
            "selected_features": "models/selected_features.json"
        },
        "data_source": {
            "existing_data": "data/train.csv"
        },
        "feature_engineering": {
            "enabled": True
        }
    }


@pytest.fixture
def sample_input_data():
    """Sample input data for testing"""
    return pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "feature1": [10, 20, 30, 40],
        "feature2": [100, 200, 300, 400],
        "feature3": ["A", "B", "C", "D"]
    })


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing"""
    return np.array([150000, 250000, 350000, 450000])


def test_run_inference_with_fixtures(sample_config, sample_input_data, sample_predictions):
    """Test using fixtures"""
    with patch(f'{MODULE_PATH}.logger'):
        with patch('builtins.open', mock_open()):
            with patch(f'{MODULE_PATH}.yaml.safe_load', return_value=sample_config):
                with patch(f'{MODULE_PATH}.pd.read_csv', return_value=sample_input_data):
                    with patch(f'{MODULE_PATH}.engineer_features', return_value=sample_input_data):
                        with patch(f'{MODULE_PATH}.transform_with_pipeline', return_value=sample_input_data):
                            with patch(f'{MODULE_PATH}.pickle.load') as mock_pickle:
                                with patch(f'{MODULE_PATH}.pd.read_json') as mock_read_json:
                                    with patch('pathlib.Path'):
                                        with patch(f'{MODULE_PATH}.pd.DataFrame') as mock_df:
                                            
                                            # Setup mocks
                                            mock_model = MagicMock()
                                            mock_model.predict.return_value = sample_predictions
                                            mock_pipeline = MagicMock()
                                            mock_pickle.side_effect = [mock_pipeline, mock_model]
                                            mock_read_json.return_value.tolist.return_value = ["feature1", "feature2"]
                                            mock_result_df = MagicMock()
                                            mock_df.return_value = mock_result_df
                                            
                                            # Run inference
                                            run_inference("input.csv", "config.yaml", "output.csv")
                                            
                                            # Verify predictions DataFrame was created correctly
                                            mock_df.assert_called_once_with({"prediction": sample_predictions})
                                            mock_result_df.to_csv.assert_called_once_with("output.csv", index=False)


# Test runner configuration
if __name__ == "__main__":
    # Configure for verbose testing
    pytest.main(["-v", __file__])