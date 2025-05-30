import os
import pytest
import pandas as pd
import yaml
import tempfile
import logging
from unittest.mock import patch, mock_open, MagicMock
from io import StringIO

# Import the module under test - UPDATE THIS LINE BASED ON YOUR PROJECT STRUCTURE
import sys
import os
# Add the parent directory to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.data_load.data_loader import load_config, load_env, load_data, get_data
except ImportError:
    # Fallback: try importing from current directory
    from data_loader import load_config, load_env, load_data, get_data


class TestLoadConfig:
    """Test cases for load_config function"""
    
    def test_load_config_success(self):
        """Test successful config loading"""
        config_data = {
            "data_source": {
                "raw_path": "data/raw.csv",
                "processed_path": "data/processed.csv",
                "type": "csv"
            }
        }
        yaml_content = yaml.dump(config_data)
        
        with patch("os.path.isfile", return_value=True):
            with patch("builtins.open", mock_open(read_data=yaml_content)):
                result = load_config("config.yaml")
                
        assert result == config_data
    
    def test_load_config_file_not_found(self):
        """Test FileNotFoundError when config file doesn't exist"""
        with patch("os.path.isfile", return_value=False):
            with pytest.raises(FileNotFoundError, match="Config file not found: nonexistent.yaml"):
                load_config("nonexistent.yaml")
    
    def test_load_config_invalid_yaml(self):
        """Test handling of invalid YAML content"""
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch("os.path.isfile", return_value=True):
            with patch("builtins.open", mock_open(read_data=invalid_yaml)):
                with pytest.raises(yaml.YAMLError):
                    load_config("config.yaml")


class TestLoadEnv:
    """Test cases for load_env function"""
    
    @patch('src.data_load.data_loader.load_dotenv')
    @patch('src.data_load.data_loader.logger')
    def test_load_env_file_exists(self, mock_logger, mock_load_dotenv):
        """Test successful .env file loading"""
        with patch("os.path.isfile", return_value=True):
            load_env(".env")
            
        mock_load_dotenv.assert_called_once_with(dotenv_path=".env", override=True)
        mock_logger.info.assert_called_once_with("Environment variables loaded from .env")
    
    @patch('src.data_load.data_loader.logger')
    def test_load_env_file_not_exists(self, mock_logger):
        """Test warning when .env file doesn't exist"""
        with patch("os.path.isfile", return_value=False):
            load_env(".env")
            
        mock_logger.warning.assert_called_once_with("No .env file found at .env")


class TestLoadData:
    """Test cases for load_data function"""
    
    @patch('src.data_load.data_loader.pd.read_csv')
    @patch('src.data_load.data_loader.logger')
    def test_load_data_csv_success(self, mock_logger, mock_read_csv):
        """Test successful CSV loading"""
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = mock_df
        
        with patch("os.path.isfile", return_value=True):
            result = load_data("test.csv", file_type="csv")
            
        mock_read_csv.assert_called_once_with(
            "test.csv", delimiter=",", header=0, encoding="utf-8"
        )
        mock_logger.info.assert_called_once_with("Loaded data from test.csv, shape=(2, 2)")
        pd.testing.assert_frame_equal(result, mock_df)
    
    @patch('src.data_load.data_loader.pd.read_excel')
    @patch('src.data_load.data_loader.logger')
    def test_load_data_excel_success(self, mock_logger, mock_read_excel):
        """Test successful Excel loading"""
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_excel.return_value = mock_df
        
        with patch("os.path.isfile", return_value=True):
            result = load_data("test.xlsx", file_type="excel", sheet_name="Sheet1")
            
        mock_read_excel.assert_called_once_with(
            "test.xlsx", sheet_name="Sheet1", header=0, engine="openpyxl"
        )
        mock_logger.info.assert_called_once_with("Loaded data from test.xlsx, shape=(2, 2)")
        pd.testing.assert_frame_equal(result, mock_df)
    
    @patch('src.data_load.data_loader.pd.read_excel')
    def test_load_data_excel_multiple_sheets(self, mock_read_excel):
        """Test Excel loading with multiple sheets (should raise ValueError)"""
        mock_read_excel.return_value = {"Sheet1": pd.DataFrame(), "Sheet2": pd.DataFrame()}
        
        with patch("os.path.isfile", return_value=True):
            with pytest.raises(ValueError, match="Multiple sheets detected"):
                load_data("test.xlsx", file_type="excel")
    
    def test_load_data_invalid_path(self):
        """Test ValueError for invalid path"""
        with pytest.raises(ValueError, match="Invalid path provided"):
            load_data("")
        
        with pytest.raises(ValueError, match="Invalid path provided"):
            load_data(None)
    
    def test_load_data_file_not_found(self):
        """Test FileNotFoundError when file doesn't exist"""
        with patch("os.path.isfile", return_value=False):
            with pytest.raises(FileNotFoundError, match="Data file not found: nonexistent.csv"):
                load_data("nonexistent.csv")
    
    def test_load_data_unsupported_file_type(self):
        """Test ValueError for unsupported file type"""
        with patch("os.path.isfile", return_value=True):
            with pytest.raises(ValueError, match="Unsupported file type: json"):
                load_data("test.json", file_type="json")
    
    @patch('src.data_load.data_loader.pd.read_csv')
    @patch('src.data_load.data_loader.logger')
    def test_load_data_pandas_exception(self, mock_logger, mock_read_csv):
        """Test exception handling when pandas read fails"""
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No data")
        
        with patch("os.path.isfile", return_value=True):
            with pytest.raises(pd.errors.EmptyDataError):
                load_data("empty.csv")
        
        mock_logger.exception.assert_called_once_with("Data loading failed.")


class TestGetData:
    """Test cases for get_data function"""
    
    @patch('src.data_load.data_loader.load_data')
    @patch('src.data_load.data_loader.load_config')
    @patch('src.data_load.data_loader.load_env')
    def test_get_data_raw_success(self, mock_load_env, mock_load_config, mock_load_data):
        """Test successful raw data loading"""
        mock_config = {
            "data_source": {
                "raw_path": "data/raw.csv",
                "processed_path": "data/processed.csv",
                "type": "csv",
                "delimiter": ",",
                "header": 0,
                "encoding": "utf-8"
            }
        }
        mock_load_config.return_value = mock_config
        mock_df = pd.DataFrame({"col1": [1, 2]})
        mock_load_data.return_value = mock_df
        
        result = get_data("config.yaml", ".env", "raw")
        
        mock_load_env.assert_called_once_with(".env")
        mock_load_config.assert_called_once_with("config.yaml")
        mock_load_data.assert_called_once_with(
            path="data/raw.csv",
            file_type="csv",
            sheet_name=None,
            delimiter=",",
            header=0,
            encoding="utf-8"
        )
        pd.testing.assert_frame_equal(result, mock_df)
    
    @patch('src.data_load.data_loader.load_data')
    @patch('src.data_load.data_loader.load_config')
    @patch('src.data_load.data_loader.load_env')
    def test_get_data_processed_success(self, mock_load_env, mock_load_config, mock_load_data):
        """Test successful processed data loading"""
        mock_config = {
            "data_source": {
                "raw_path": "data/raw.csv",
                "processed_path": "data/processed.csv",
                "type": "excel",
                "sheet_name": "Data"
            }
        }
        mock_load_config.return_value = mock_config
        mock_df = pd.DataFrame({"col1": [1, 2]})
        mock_load_data.return_value = mock_df
        
        result = get_data("config.yaml", ".env", "processed")
        
        mock_load_data.assert_called_once_with(
            path="data/processed.csv",
            file_type="excel",
            sheet_name="Data",
            delimiter=",",
            header=0,
            encoding="utf-8"
        )
        pd.testing.assert_frame_equal(result, mock_df)
    
    @patch('src.data_load.data_loader.load_config')
    @patch('src.data_load.data_loader.load_env')
    def test_get_data_unknown_stage(self, mock_load_env, mock_load_config):
        """Test ValueError for unknown data_stage"""
        mock_config = {"data_source": {}}
        mock_load_config.return_value = mock_config
        
        with pytest.raises(ValueError, match="Unknown data_stage: invalid"):
            get_data("config.yaml", ".env", "invalid")
    
    @patch('src.data_load.data_loader.load_config')
    @patch('src.data_load.data_loader.load_env')
    @patch('src.data_load.data_loader.logger')
    def test_get_data_missing_raw_path(self, mock_logger, mock_load_env, mock_load_config):
        """Test ValueError when raw_path is missing"""
        mock_config = {
            "data_source": {
                "processed_path": "data/processed.csv"
                # raw_path is missing
            }
        }
        mock_load_config.return_value = mock_config
        
        with pytest.raises(ValueError, match="Missing path for data_stage='raw'"):
            get_data("config.yaml", ".env", "raw")
        
        mock_logger.error.assert_called_with("No valid path for data_stage='raw'")
    
    @patch('src.data_load.data_loader.load_config')
    @patch('src.data_load.data_loader.load_env')
    def test_get_data_invalid_path_type(self, mock_load_env, mock_load_config):
        """Test ValueError when path is not a string"""
        mock_config = {
            "data_source": {
                "raw_path": 123,  # Not a string
                "type": "csv"
            }
        }
        mock_load_config.return_value = mock_config
        
        with pytest.raises(ValueError, match="Missing path for data_stage='raw'"):
            get_data("config.yaml", ".env", "raw")
    
    @patch('src.data_load.data_loader.load_config')
    @patch('src.data_load.data_loader.load_env')
    def test_get_data_empty_data_source(self, mock_load_env, mock_load_config):
        """Test when data_source is missing from config"""
        mock_config = {}  # No data_source key
        mock_load_config.return_value = mock_config
        
        with pytest.raises(ValueError, match="Missing path for data_stage='raw'"):
            get_data("config.yaml", ".env", "raw")


class TestIntegration:
    """Integration tests using temporary files"""
    
    def test_integration_csv_loading(self):
        """Test end-to-end CSV loading with real files"""
        # Create temporary CSV file
        csv_content = "col1,col2\n1,2\n3,4\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name
        
        # Create temporary config file
        config_data = {
            "data_source": {
                "raw_path": csv_path,
                "type": "csv",
                "delimiter": ",",
                "header": 0,
                "encoding": "utf-8"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Test the integration
            with patch('src.data_load.data_loader.load_env'):  # Skip .env loading
                df = get_data(config_path, data_stage="raw")
            
            expected_df = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
            pd.testing.assert_frame_equal(df, expected_df)
            
        finally:
            # Clean up temporary files
            os.unlink(csv_path)
            os.unlink(config_path)


# Fixtures for common test data
@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "data_source": {
            "raw_path": "data/raw.csv",
            "processed_path": "data/processed.csv",
            "type": "csv",
            "delimiter": ",",
            "header": 0,
            "encoding": "utf-8"
        }
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing"""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "value": [10.5, 20.3, 15.7]
    })


# Test runner configuration
if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Run tests with pytest
    pytest.main(["-v", __file__])