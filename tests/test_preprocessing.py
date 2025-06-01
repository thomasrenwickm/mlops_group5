"""
test_preprocessing.py

Unit tests for preprocessing.py module
"""
import pytest
import pandas as pd
import numpy as np
import sys
from unittest.mock import patch, mock_open, MagicMock
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Import the functions to test
from src.preprocess.preprocessing import (
    clean_raw_data,
    build_preprocessing_pipeline,
    fit_and_save_pipeline,
    transform_with_pipeline
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'old_name': [1, 2, 3, 4, 5],
        'category': ['A', 'B', None, 'A', 'C'],
        'numerical': [10.5, None, 30.0, 40.0, 50.0],
        'to_drop': ['x', 'y', 'z', 'w', 'v'],
        'group_col': ['G1', 'G1', 'G2', 'G2', 'G1'],
        'critical': [1, 2, 3, None, 5]
    })


@pytest.fixture
def basic_config():
    """Create a basic configuration for testing."""
    return {
        "preprocessing": {
            "rename_columns": {"old_name": "new_name"},
            "drop_columns": ["to_drop"],
            "fillna": {
                "numerical_zero": ["numerical"],
                "categorical_none": ["category"]
            },
            "fillna_groupby": {
                "column": "numerical",
                "groupby": "group_col"
            },
            "dropna_rows": ["critical"],
            "encoding": {
                "one_hot": {
                    "handle_unknown": "ignore",
                    "drop": None
                }
            },
            "scaling": {
                "method": "standard",
                "apply_to": ["continuous", "ordinal"]
            }
        },
        "features": {
            "categorical": ["category"],
            "ordinal": ["new_name"],
            "continuous": ["numerical"]
        },
        "artifacts": {
            "preprocessing_pipeline": "models/pipeline.pkl",
            "selected_features": "models/features.json"
        }
    }


class TestCleanRawData:
    """Test cases for clean_raw_data function."""
    
    def test_rename_columns(self, sample_dataframe, basic_config):
        """Test column renaming functionality."""
        result = clean_raw_data(sample_dataframe.copy(), basic_config)
        assert 'new_name' in result.columns
        assert 'old_name' not in result.columns
    
    def test_drop_columns(self, sample_dataframe, basic_config):
        """Test column dropping functionality."""
        result = clean_raw_data(sample_dataframe.copy(), basic_config)
        assert 'to_drop' not in result.columns
    
    def test_fillna_numerical_zero(self, sample_dataframe, basic_config):
        """Test filling numerical columns with zero."""
        result = clean_raw_data(sample_dataframe.copy(), basic_config)
        # Note: fillna_groupby will handle the numerical column in this config
        assert not result['numerical'].isna().any()
    
    def test_fillna_categorical_none(self, sample_dataframe, basic_config):
        """Test filling categorical columns with 'None'."""
        result = clean_raw_data(sample_dataframe.copy(), basic_config)
        assert (result['category'] == 'None').sum() == 1
    
    def test_fillna_groupby(self, sample_dataframe, basic_config):
        """Test group-based filling of missing values."""
        df = sample_dataframe.copy()
        df.loc[1, 'numerical'] = None  # Add another missing value
        result = clean_raw_data(df, basic_config)
        
        # Check that missing values were filled with group means
        assert not result['numerical'].isna().any()
    
    def test_dropna_rows(self, sample_dataframe, basic_config):
        """Test dropping rows with missing values in critical columns."""
        result = clean_raw_data(sample_dataframe.copy(), basic_config)
        assert len(result) == 4  # One row should be dropped due to missing 'critical'
        assert not result['critical'].isna().any()
    
    def test_empty_config_sections(self, sample_dataframe):
        """Test handling of empty configuration sections."""
        empty_config = {"preprocessing": {}}
        result = clean_raw_data(sample_dataframe.copy(), empty_config)
        # Should return original dataframe unchanged
        pd.testing.assert_frame_equal(result, sample_dataframe)
    
    def test_missing_columns_in_config(self, sample_dataframe, basic_config):
        """Test handling when configured columns don't exist in DataFrame."""
        config = basic_config.copy()
        config["preprocessing"]["drop_columns"] = ["nonexistent_col"]
        config["preprocessing"]["fillna"]["numerical_zero"] = ["nonexistent_numerical"]
        
        # Should not raise error and should process normally
        result = clean_raw_data(sample_dataframe.copy(), config)
        assert len(result.columns) > 0


class TestBuildPreprocessingPipeline:
    """Test cases for build_preprocessing_pipeline function."""

    
    def test_pipeline_with_missing_columns(self, basic_config):
        """Test pipeline construction when some configured columns are missing."""
        df = pd.DataFrame({'existing_col': [1, 2, 3]})
        pipeline, features = build_preprocessing_pipeline(basic_config, df)
        
        assert isinstance(pipeline, ColumnTransformer)
        assert len(features) == 0  # No configured columns exist in df
    
    def test_pipeline_transformers(self, sample_dataframe, basic_config):
        """Test that correct transformers are added to pipeline."""
        pipeline, _ = build_preprocessing_pipeline(basic_config, sample_dataframe)
        
        transformer_names = [name for name, _, _ in pipeline.transformers]
        assert 'onehot' in transformer_names
        assert 'scaler' in transformer_names
    
    def test_no_encoding_config(self, sample_dataframe, basic_config):
        """Test pipeline construction without encoding configuration."""
        config = basic_config.copy()
        del config["preprocessing"]["encoding"]
        
        pipeline, features = build_preprocessing_pipeline(config, sample_dataframe)
        transformer_names = [name for name, _, _ in pipeline.transformers]
        assert 'onehot' not in transformer_names
    
    def test_no_scaling_config(self, sample_dataframe, basic_config):
        """Test pipeline construction without scaling configuration."""
        config = basic_config.copy()
        del config["preprocessing"]["scaling"]
        
        pipeline, features = build_preprocessing_pipeline(config, sample_dataframe)
        transformer_names = [name for name, _, _ in pipeline.transformers]
        assert 'scaler' not in transformer_names
    
    def test_minmax_scaler(self, sample_dataframe, basic_config):
        """Test pipeline construction with MinMaxScaler."""
        config = basic_config.copy()
        config["preprocessing"]["scaling"]["method"] = "minmax"
        
        pipeline, _ = build_preprocessing_pipeline(config, sample_dataframe)
        
        # Find the scaler transformer
        scaler_transformer = None
        for name, transformer, _ in pipeline.transformers:
            if name == 'scaler':
                scaler_transformer = transformer
                break
        
        assert scaler_transformer is not None
        assert scaler_transformer.__class__.__name__ == 'MinMaxScaler'


class TestFitAndSavePipeline:
    """Test cases for fit_and_save_pipeline function."""
    
    @patch('src.preprocess.preprocessing.os.makedirs')
    @patch('src.preprocess.preprocessing.pd.Series.to_json')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.preprocess.preprocessing.pickle.dump')
    def test_fit_and_save_pipeline(self, mock_pickle_dump, mock_file, mock_to_json, 
                                   mock_makedirs, sample_dataframe, basic_config):
        """Test fitting and saving pipeline functionality."""
        with patch('src.preprocess.preprocessing.logger'):
            result = fit_and_save_pipeline(sample_dataframe.copy(), basic_config)
        
        # Check that directories are created
        mock_makedirs.assert_called_once()
        
        # Check that pipeline is saved
        mock_pickle_dump.assert_called_once()
        mock_file.assert_called()
        
        # Check that features are saved
        mock_to_json.assert_called_once()
        
        # Check return value
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    @patch('src.preprocess.preprocessing.os.makedirs')
    @patch('src.preprocess.preprocessing.pd.Series.to_json')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.preprocess.preprocessing.pickle.dump')
    def test_feature_names_fallback(self, mock_pickle_dump, mock_file, 
                                    mock_to_json, mock_makedirs, 
                                    sample_dataframe, basic_config):
        """Test fallback feature naming when get_feature_names_out fails."""
        with patch('src.preprocess.preprocessing.logger'):
            with patch('src.preprocess.preprocessing.build_preprocessing_pipeline') as mock_build:
                # Create a mock pipeline that doesn't have get_feature_names_out
                mock_pipeline = MagicMock()
                mock_pipeline.fit_transform.return_value = np.array([[1, 2], [3, 4]])
                del mock_pipeline.get_feature_names_out  # Simulate AttributeError
                mock_build.return_value = (mock_pipeline, ['col1', 'col2'])
                
                result = fit_and_save_pipeline(sample_dataframe.copy(), basic_config)
                
                # Should use fallback feature names
                expected_columns = ['f0', 'f1']
                assert list(result.columns) == expected_columns


class TestTransformWithPipeline:
    """Test cases for transform_with_pipeline function."""
    
    def test_transform_with_pipeline(self, sample_dataframe, basic_config):
        """Test transforming data with a fitted pipeline."""
        # Create a mock fitted pipeline
        mock_pipeline = MagicMock()
        transformed_data = np.array([[1, 0, 0, 2.5], [0, 1, 0, 1.5]])
        mock_pipeline.transform.return_value = transformed_data
        mock_pipeline.get_feature_names_out.return_value = ['cat_A', 'cat_B', 'cat_C', 'num_scaled']
        
        with patch('src.preprocess.preprocessing.logger'):
            result = transform_with_pipeline(sample_dataframe.copy(), basic_config, mock_pipeline)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['cat_A', 'cat_B', 'cat_C', 'num_scaled']
        assert len(result) == 2
    
    def test_transform_feature_names_fallback(self, sample_dataframe, basic_config):
        """Test fallback feature naming during transformation."""
        # Create a mock pipeline without get_feature_names_out
        mock_pipeline = MagicMock()
        transformed_data = np.array([[1, 2], [3, 4]])
        mock_pipeline.transform.return_value = transformed_data
        del mock_pipeline.get_feature_names_out  # Simulate AttributeError
        
        with patch('src.preprocess.preprocessing.logger'):
            result = transform_with_pipeline(sample_dataframe.copy(), basic_config, mock_pipeline)
        
        expected_columns = ['f0', 'f1']
        assert list(result.columns) == expected_columns


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    @patch('src.preprocess.preprocessing.os.makedirs')
    @patch('src.preprocess.preprocessing.pd.Series.to_json')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.preprocess.preprocessing.pickle.dump')
    def test_full_preprocessing_workflow(self, mock_pickle_dump, mock_file, 
                                         mock_to_json, mock_makedirs, 
                                         sample_dataframe, basic_config):
        """Test the complete preprocessing workflow."""
        with patch('src.preprocess.preprocessing.logger'):
            # Fit and save pipeline
            processed_df = fit_and_save_pipeline(sample_dataframe.copy(), basic_config)
            
            # Verify the processed data
            assert isinstance(processed_df, pd.DataFrame)
            assert len(processed_df) == 4  # One row dropped due to missing critical value
            
            # All functions should have been called
            mock_makedirs.assert_called_once()
            mock_pickle_dump.assert_called_once()
            mock_to_json.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])