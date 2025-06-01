"""
test_data_loader.py

Simple unit tests for data_loader.py module
"""
import pytest
import pandas as pd
import os
from unittest.mock import patch, mock_open, MagicMock

# Import the functions to test
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_load.data_loader import load_config, load_env, load_data, get_data


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "data_source": {
            "raw_path": "data/raw/Ames_Housing_Data.csv",
            "processed_path": "data/processed/processed_data.csv",
            "type": "csv",
            "delimiter": ",",
            "header": 0,
            "encoding": "utf-8"
        }
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'C', 'D', 'E'],
        'target': [100, 200, 300, 400, 500]
    })


class TestLoadConfig:
    """Test cases for load_config function."""
    
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.isfile')
    def test_load_config_success(self, mock_isfile, mock_file, mock_yaml, sample_config):
        """Test successful config loading."""
        mock_isfile.return_value = True
        mock_yaml.return_value = sample_config
        
        result = load_config("config.yaml")
        
        assert result == sample_config
        mock_isfile.assert_called_once_with("config.yaml")
        mock_file.assert_called_once_with("config.yaml", "r", encoding="utf-8")
        mock_yaml.assert_called_once()
    
    @patch('os.path.isfile')
    def test_load_config_file_not_found(self, mock_isfile):
        """Test config loading when file doesn't exist."""
        mock_isfile.return_value = False
        
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("missing.yaml")
    
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.isfile')
    def test_load_config_yaml_error(self, mock_isfile, mock_file, mock_yaml):
        """Test config loading with invalid YAML."""
        mock_isfile.return_value = True
        mock_yaml.side_effect = Exception("Invalid YAML")
        
        with pytest.raises(Exception):
            load_config("bad_config.yaml")


class TestLoadEnv:
    """Test cases for load_env function."""
    
    @patch('src.data_load.data_loader.logger')
    @patch('src.data_load.data_loader.load_dotenv')
    @patch('os.path.isfile')
    def test_load_env_file_exists(self, mock_isfile, mock_load_dotenv, mock_logger):
        """Test loading environment when .env file exists."""
        mock_isfile.return_value = True
        
        load_env(".env")
        
        mock_isfile.assert_called_once_with(".env")
        mock_load_dotenv.assert_called_once_with(dotenv_path=".env", override=True)
        mock_logger.info.assert_called_once()
    
    @patch('src.data_load.data_loader.logger')
    @patch('os.path.isfile')
    def test_load_env_file_missing(self, mock_isfile, mock_logger):
        """Test loading environment when .env file doesn't exist."""
        mock_isfile.return_value = False
        
        load_env(".env")
        
        mock_isfile.assert_called_once_with(".env")
        mock_logger.warning.assert_called_once()


class TestLoadData:
    """Test cases for load_data function."""
    
    @patch('src.data_load.data_loader.logger')
    @patch('pandas.read_csv')
    @patch('os.path.isfile')
    def test_load_data_csv_success(self, mock_isfile, mock_read_csv, mock_logger, sample_dataframe):
        """Test successful CSV data loading."""
        mock_isfile.return_value = True
        mock_read_csv.return_value = sample_dataframe
        
        result = load_data("test.csv", file_type="csv")
        
        assert result.equals(sample_dataframe)
        mock_isfile.assert_called_once_with("test.csv")
        mock_read_csv.assert_called_once_with(
            "test.csv", delimiter=",", header=0, encoding="utf-8"
        )
        mock_logger.info.assert_called_once()
    
    @patch('src.data_load.data_loader.logger')
    @patch('pandas.read_excel')
    @patch('os.path.isfile')
    def test_load_data_excel_success(self, mock_isfile, mock_read_excel, mock_logger, sample_dataframe):
        """Test successful Excel data loading."""
        mock_isfile.return_value = True
        mock_read_excel.return_value = sample_dataframe
        
        result = load_data("test.xlsx", file_type="excel", sheet_name="Sheet1")
        
        assert result.equals(sample_dataframe)
        mock_isfile.assert_called_once_with("test.xlsx")
        mock_read_excel.assert_called_once_with(
            "test.xlsx", sheet_name="Sheet1", header=0, engine="openpyxl"
        )
    
    def test_load_data_invalid_path(self):
        """Test load_data with invalid path."""
        with pytest.raises(ValueError, match="Invalid path provided"):
            load_data("")
        
        with pytest.raises(ValueError, match="Invalid path provided"):
            load_data(None)
    
    @patch('os.path.isfile')
    def test_load_data_file_not_found(self, mock_isfile):
        """Test load_data when file doesn't exist."""
        mock_isfile.return_value = False
        
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_data("missing.csv")
    
    @patch('os.path.isfile')
    def test_load_data_unsupported_file_type(self, mock_isfile):
        """Test load_data with unsupported file type."""
        mock_isfile.return_value = True
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_data("test.txt", file_type="txt")
    
    @patch('src.data_load.data_loader.logger')
    @patch('pandas.read_excel')
    @patch('os.path.isfile')
    def test_load_data_excel_multiple_sheets(self, mock_isfile, mock_read_excel, mock_logger):
        """Test Excel loading with multiple sheets error."""
        mock_isfile.return_value = True
        mock_read_excel.return_value = {"Sheet1": pd.DataFrame(), "Sheet2": pd.DataFrame()}
        
        with pytest.raises(ValueError, match="Multiple sheets detected"):
            load_data("test.xlsx", file_type="excel")
    
    @patch('src.data_load.data_loader.logger')
    @patch('pandas.read_csv')
    @patch('os.path.isfile')
    def test_load_data_read_error(self, mock_isfile, mock_read_csv, mock_logger):
        """Test load_data when pandas read fails."""
        mock_isfile.return_value = True
        mock_read_csv.side_effect = Exception("Read error")
        
        with pytest.raises(Exception):
            load_data("test.csv")
        
        mock_logger.exception.assert_called_once_with("Data loading failed.")


class TestGetData:
    """Test cases for get_data function."""
    
    @patch('src.data_load.data_loader.load_data')
    @patch('src.data_load.data_loader.load_config')
    @patch('src.data_load.data_loader.load_env')
    def test_get_data_raw_success(self, mock_load_env, mock_load_config, mock_load_data, 
                                  sample_config, sample_dataframe):
        """Test successful raw data retrieval."""
        mock_load_config.return_value = sample_config
        mock_load_data.return_value = sample_dataframe
        
        result = get_data(data_stage="raw")
        
        assert result.equals(sample_dataframe)
        mock_load_env.assert_called_once_with(".env")
        mock_load_config.assert_called_once_with("config.yaml")
        mock_load_data.assert_called_once_with(
            path="data/raw/Ames_Housing_Data.csv",
            file_type="csv",
            sheet_name=None,
            delimiter=",",
            header=0,
            encoding="utf-8"
        )
    
    @patch('src.data_load.data_loader.load_data')
    @patch('src.data_load.data_loader.load_config')
    @patch('src.data_load.data_loader.load_env')
    def test_get_data_processed_success(self, mock_load_env, mock_load_config, mock_load_data,
                                        sample_config, sample_dataframe):
        """Test successful processed data retrieval."""
        mock_load_config.return_value = sample_config
        mock_load_data.return_value = sample_dataframe
        
        result = get_data(data_stage="processed")
        
        assert result.equals(sample_dataframe)
        mock_load_data.assert_called_once_with(
            path="data/processed/processed_data.csv",
            file_type="csv",
            sheet_name=None,
            delimiter=",",
            header=0,
            encoding="utf-8"
        )
    
    @patch('src.data_load.data_loader.logger')
    @patch('src.data_load.data_loader.load_config')
    @patch('src.data_load.data_loader.load_env')
    def test_get_data_invalid_stage(self, mock_load_env, mock_load_config, mock_logger, sample_config):
        """Test get_data with invalid data_stage."""
        mock_load_config.return_value = sample_config
        
        with pytest.raises(ValueError, match="Unknown data_stage"):
            get_data(data_stage="invalid")
        
        mock_logger.error.assert_called_once()
    
    @patch('src.data_load.data_loader.logger')
    @patch('src.data_load.data_loader.load_config')
    @patch('src.data_load.data_loader.load_env')
    def test_get_data_missing_path(self, mock_load_env, mock_load_config, mock_logger):
        """Test get_data when config missing required path."""
        # Config without raw_path
        config_no_path = {
            "data_source": {
                "processed_path": "data/processed/processed_data.csv"
            }
        }
        mock_load_config.return_value = config_no_path
        
        with pytest.raises(ValueError, match="Missing path for data_stage"):
            get_data(data_stage="raw")
        
        mock_logger.error.assert_called_once()
    
    @patch('src.data_load.data_loader.load_data')
    @patch('src.data_load.data_loader.load_config')
    @patch('src.data_load.data_loader.load_env')
    def test_get_data_custom_paths(self, mock_load_env, mock_load_config, mock_load_data, sample_dataframe):
        """Test get_data with custom config and env paths."""
        config = {
            "data_source": {
                "raw_path": "custom/path/data.csv",
                "type": "csv"
            }
        }
        mock_load_config.return_value = config
        mock_load_data.return_value = sample_dataframe
        
        result = get_data(config_path="custom_config.yaml", env_path="custom.env", data_stage="raw")
        
        mock_load_env.assert_called_once_with("custom.env")
        mock_load_config.assert_called_once_with("custom_config.yaml")
        assert result.equals(sample_dataframe)


class TestIntegration:
    """Integration tests for the data_loader module."""
    
    @patch('src.data_load.data_loader.load_dotenv')
    @patch('pandas.read_csv')
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.isfile')
    def test_full_workflow(self, mock_isfile, mock_file, mock_yaml, mock_read_csv, 
                           mock_load_dotenv, sample_config, sample_dataframe):
        """Test complete workflow from config to data loading."""
        # Setup mocks
        mock_isfile.return_value = True
        mock_yaml.return_value = sample_config
        mock_read_csv.return_value = sample_dataframe
        
        # Execute full workflow
        result = get_data(data_stage="raw")
        
        # Verify workflow steps
        assert result.equals(sample_dataframe)
        mock_yaml.assert_called_once()
        mock_read_csv.assert_called_once()
        assert mock_isfile.call_count >= 1  # Called for both config and data files


if __name__ == "__main__":
    pytest.main([__file__])