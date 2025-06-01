"""
test_inference.py

Simple unit tests for inference.py module
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

# Import the function to test
import sys
import os

# Add both project root and src to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')

# Add both paths so imports work correctly
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from inference.inference import run_inference


@pytest.fixture
def sample_config():
    """Basic config for testing."""
    return {
        "data_source": {"new_data_path": ""},
        "artifacts": {
            "model": "model.pkl",
            "preprocessing_pipeline": "pipeline.pkl",
            "selected_features": "features.json",
            "inference_output": ""
        }
    }


@pytest.fixture
def sample_data():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [0.1, 0.2, 0.3]
    })


class TestRunInference:
    """Test cases for run_inference function."""
    
    @patch('inference.inference.logger')
    @patch('pandas.DataFrame')
    @patch('inference.inference.Path')
    @patch('pickle.load')
    @patch('pandas.read_json')
    @patch('inference.inference.transform_with_pipeline')
    @patch('inference.inference.engineer_features')
    @patch('pandas.read_csv')
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_inference_success(self, mock_file, mock_yaml, mock_read_csv, 
                                   mock_engineer, mock_transform, mock_read_json,
                                   mock_pickle, mock_path, mock_dataframe, mock_logger,
                                   sample_config, sample_data):
        """Test successful inference execution."""
        
        # Setup mocks
        mock_yaml.return_value = sample_config
        mock_read_csv.return_value = sample_data
        mock_engineer.return_value = sample_data
        mock_transform.return_value = sample_data
        mock_read_json.return_value = pd.Series(['feature1', 'feature2'])
        
        # Mock model and pipeline
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.8, 0.3])
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_pipeline, mock_model]
        
        # Mock Path operations
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_path_obj.parent.mkdir = MagicMock()
        
        # Mock DataFrame creation and to_csv
        mock_df_instance = MagicMock()
        mock_dataframe.return_value = mock_df_instance
        
        # Run the function
        run_inference("input.csv", "config.yaml", "output.csv")
        
        # Verify key operations happened
        mock_yaml.assert_called_once()
        mock_read_csv.assert_called_once_with("input.csv")
        mock_engineer.assert_called_once()
        mock_transform.assert_called_once()
        mock_model.predict.assert_called_once()
        mock_df_instance.to_csv.assert_called_once_with("output.csv", index=False)
        mock_path_obj.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @patch('inference.inference.logger')
    @patch('pandas.DataFrame')
    @patch('inference.inference.Path')
    @patch('pickle.load')
    @patch('pandas.read_json')
    @patch('inference.inference.transform_with_pipeline')
    @patch('inference.inference.engineer_features')
    @patch('pandas.read_csv')
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    def test_feature_selection(self, mock_file, mock_yaml, mock_read_csv,
                               mock_engineer, mock_transform, mock_read_json,
                               mock_pickle, mock_path, mock_dataframe, mock_logger,
                               sample_config, sample_data):
        """Test that only selected features are used."""
        
        # Setup mocks
        mock_yaml.return_value = sample_config
        mock_read_csv.return_value = sample_data
        mock_engineer.return_value = sample_data
        mock_transform.return_value = sample_data
        
        # Only select one feature
        mock_read_json.return_value = pd.Series(['feature1'])
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.6, 0.7])
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_pipeline, mock_model]
        
        # Mock Path and DataFrame
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_df_instance = MagicMock()
        mock_dataframe.return_value = mock_df_instance
        
        # Run the function
        run_inference("input.csv", "config.yaml", "output.csv")
        
        # Verify that model.predict was called
        mock_model.predict.assert_called_once()
        # Verify that the model received data with selected features
        predict_call_data = mock_model.predict.call_args[0][0]
        assert list(predict_call_data.columns) == ['feature1']
    
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    def test_file_not_found(self, mock_file, mock_yaml):
        """Test handling of missing files."""
        mock_yaml.return_value = {}
        mock_file.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            run_inference("missing.csv", "config.yaml", "output.csv")
    
    @patch('inference.inference.logger')
    @patch('pandas.DataFrame')
    @patch('inference.inference.Path')
    @patch('pickle.load')
    @patch('pandas.read_json')
    @patch('inference.inference.transform_with_pipeline')
    @patch('inference.inference.engineer_features')
    @patch('pandas.read_csv')
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    def test_config_update(self, mock_file, mock_yaml, mock_read_csv,
                           mock_engineer, mock_transform, mock_read_json,
                           mock_pickle, mock_path, mock_dataframe, mock_logger,
                           sample_config, sample_data):
        """Test that config is updated with file paths."""
        
        # Setup mocks
        config_copy = sample_config.copy()
        mock_yaml.return_value = config_copy
        mock_read_csv.return_value = sample_data
        mock_engineer.return_value = sample_data
        mock_transform.return_value = sample_data
        mock_read_json.return_value = pd.Series(['feature1'])
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1])
        mock_pipeline = MagicMock()
        mock_pickle.side_effect = [mock_pipeline, mock_model]
        
        # Mock Path and DataFrame
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_df_instance = MagicMock()
        mock_dataframe.return_value = mock_df_instance
        
        # Run with specific paths
        input_path = "test_input.csv"
        output_path = "test_output.csv"
        run_inference(input_path, "config.yaml", output_path)
        
        # Check that engineer_features was called with updated config
        engineer_call_config = mock_engineer.call_args[0][1]
        assert engineer_call_config["data_source"]["new_data_path"] == input_path
        assert engineer_call_config["artifacts"]["inference_output"] == output_path


if __name__ == "__main__":
    pytest.main([__file__])