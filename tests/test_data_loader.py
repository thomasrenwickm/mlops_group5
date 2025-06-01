import os
import sys
import pytest
import pandas as pd
import yaml
from unittest.mock import patch, mock_open, MagicMock, call
from pathlib import Path
import sys
import os


# === ✅ Add project root to PYTHONPATH (assumes src/ is in root) ===
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# === ✅ Now import from your module ===
from data_load.data_loader import get_data, load_config, load_data, load_env


class TestLoadConfig:
    """Test load_config function"""
    
    @patch('data_loader.os.path.isfile')
    @patch('builtins.open', new_callable=mock_open)
    @patch('data_loader.yaml.safe_load')
    def test_load_config_success(self, mock_yaml_load, mock_file, mock_isfile):
        """Test successful config loading"""
        # Setup
        mock_isfile.return_value = True
        expected_config = {"data_source": {"raw_path": "data.csv"}}
        mock_yaml_load.return_value = expected_config
        
        # Execute
        result = load_config("config.yaml")
        
        # Assert
        mock_isfile.assert_called_once_with("config.yaml")
        mock_file.assert_called_once_with("config.yaml", "r", encoding="utf-8")
        mock_yaml_load.assert_called_once()
        assert result == expected_config
    
    @patch('data_loader.os.path.isfile')
    def test_load_config_file_not_found(self, mock_isfile):
        """Test config file not found"""
        # Setup
        mock_isfile.return_value = False
        
        # Execute & Assert
        with pytest.raises(FileNotFoundError, match="Config file not found: config.yaml"):
            load_config("config.yaml")
        
        mock_isfile.assert_called_once_with("config.yaml")
    
    @patch('data_loader.os.path.isfile')
    @patch('builtins.open', new_callable=mock_open)
    @patch('data_loader.yaml.safe_load')
    def test_load_config_yaml_error(self, mock_yaml_load, mock_file, mock_isfile):
        """Test YAML parsing error"""
        # Setup
        mock_isfile.return_value = True
        mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")
        
        # Execute & Assert
        with pytest.raises(yaml.YAMLError):
            load_config("config.yaml")


class TestLoadEnv:
    """Test load_env function"""
    
    @patch('data_loader.os.path.isfile')
    @patch('data_loader.load_dotenv')
    @patch('data_loader.logger')
    def test_load_env_file_exists(self, mock_logger, mock_load_dotenv, mock_isfile):
        """Test load_env when .env file exists"""
        # Setup
        mock_isfile.return_value = True
        env_path = ".env"
        
        # Execute
        load_env(env_path)
        
        # Assert
        mock_isfile.assert_called_once_with(env_path)
        mock_load_dotenv.assert_called_once_with(dotenv_path=env_path, override=True)
        mock_logger.info.assert_called_once_with(
            "Environment variables loaded from %s", env_path
        )
    
    @patch('data_loader.os.path.isfile')
    @patch('data_loader.load_dotenv')
    @patch('data_loader.logger')
    def test_load_env_file_not_exists(self, mock_logger, mock_load_dotenv, mock_isfile):
        """Test load_env when .env file doesn't exist"""
        # Setup
        mock_isfile.return_value = False
        env_path = ".env"
        
        # Execute
        load_env(env_path)
        
        # Assert
        mock_isfile.assert_called_once_with(env_path)
        mock_load_dotenv.assert_not_called()
        mock_logger.warning.assert_called_once_with(
            "No .env file found at %s", env_path
        )
    
    def test_load_env_default_path(self):
        """Test load_env with default path"""
        with patch('data_loader.os.path.isfile') as mock_isfile:
            mock_isfile.return_value = False
            with patch('data_loader.logger'):
                load_env()  # Should use default ".env"
            mock_isfile.assert_called_once_with(".env")


class TestLoadData:
    """Test load_data function"""
    
    @patch('data_loader.os.path.isfile')
    @patch('data_loader.pd.read_csv')
    @patch('data_loader.logger')
    def test_load_data_csv_success(self, mock_logger, mock_read_csv, mock_isfile):
        """Test successful CSV loading"""
        # Setup
        mock_isfile.return_value = True
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = mock_df
        file_path = "data.csv"
        
        # Execute
        result = load_data(file_path)
        
        # Assert
        mock_isfile.assert_called_once_with(file_path)
        mock_read_csv.assert_called_once_with(
            file_path, delimiter=",", header=0, encoding="utf-8"
        )
        pd.testing.assert_frame_equal(result, mock_df)
        mock_logger.info.assert_called_once_with(
            "Loaded data from %s, shape=%s", file_path, mock_df.shape
        )
    
    @patch('data_loader.os.path.isfile')
    @patch('data_loader.pd.read_excel')
    @patch('data_loader.logger')
    def test_load_data_excel_success(self, mock_logger, mock_read_excel, mock_isfile):
        """Test successful Excel loading"""
        # Setup
        mock_isfile.return_value = True
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_excel.return_value = mock_df
        file_path = "data.xlsx"
        
        # Execute
        result = load_data(file_path, file_type="excel", sheet_name="Sheet1")
        
        # Assert
        mock_isfile.assert_called_once_with(file_path)
        mock_read_excel.assert_called_once_with(
            file_path, sheet_name="Sheet1", header=0, engine="openpyxl"
        )
        pd.testing.assert_frame_equal(result, mock_df)
        mock_logger.info.assert_called_once_with(
            "Loaded data from %s, shape=%s", file_path, mock_df.shape
        )
    
    @patch('data_loader.os.path.isfile')
    @patch('data_loader.pd.read_excel')
    @patch('data_loader.logger')
    def test_load_data_excel_multiple_sheets_error(self, mock_logger, mock_read_excel, mock_isfile):
        """Test Excel loading with multiple sheets error"""
        # Setup
        mock_isfile.return_value = True
        mock_read_excel.return_value = {"Sheet1": pd.DataFrame(), "Sheet2": pd.DataFrame()}
        file_path = "data.xlsx"
        
        # Execute & Assert
        with pytest.raises(ValueError, match="Multiple sheets detected"):
            load_data(file_path, file_type="excel")
        
        mock_logger.exception.assert_called_once_with("Data loading failed.")
    
    @patch('data_loader.logger')
    def test_load_data_invalid_path_none(self, mock_logger):
        """Test load_data with None path"""
        # Execute & Assert
        with pytest.raises(ValueError, match="Invalid path provided"):
            load_data(None)
        
        mock_logger.error.assert_called_once_with("No valid data path specified.")
    
    @patch('data_loader.logger')
    def test_load_data_invalid_path_empty(self, mock_logger):
        """Test load_data with empty path"""
        # Execute & Assert
        with pytest.raises(ValueError, match="Invalid path provided"):
            load_data("")
        
        mock_logger.error.assert_called_once_with("No valid data path specified.")
    
    @patch('data_loader.os.path.isfile')
    @patch('data_loader.logger')
    def test_load_data_file_not_found(self, mock_logger, mock_isfile):
        """Test load_data with non-existent file"""
        # Setup
        mock_isfile.return_value = False
        file_path = "nonexistent.csv"
        
        # Execute & Assert
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_data(file_path)
        
        mock_logger.error.assert_called_once_with("File not found: %s", file_path)
    
    @patch('data_loader.os.path.isfile')
    @patch('data_loader.logger')
    def test_load_data_unsupported_file_type(self, mock_logger, mock_isfile):
        """Test load_data with unsupported file type"""
        # Setup
        mock_isfile.return_value = True
        file_path = "data.txt"
        
        # Execute & Assert
        with pytest.raises(ValueError, match="Unsupported file type: txt"):
            load_data(file_path, file_type="txt")
        
        mock_logger.exception.assert_called_once_with("Data loading failed.")
    
    @patch('data_loader.os.path.isfile')
    @patch('data_loader.pd.read_csv')
    @patch('data_loader.logger')
    def test_load_data_custom_parameters(self, mock_logger, mock_read_csv, mock_isfile):
        """Test load_data with custom parameters"""
        # Setup
        mock_isfile.return_value = True
        mock_df = pd.DataFrame({"col1": [1, 2]})
        mock_read_csv.return_value = mock_df
        file_path = "data.csv"
        
        # Execute
        result = load_data(
            file_path,
            delimiter=";",
            header=1,
            encoding="latin-1"
        )
        
        # Assert
        mock_read_csv.assert_called_once_with(
            file_path, delimiter=";", header=1, encoding="latin-1"
        )
        pd.testing.assert_frame_equal(result, mock_df)


class TestGetData:
    """Test get_data function"""
    
    @patch('data_loader.load_env')
    @patch('data_loader.load_config')
    @patch('data_loader.load_data')
    def test_get_data_raw_success(self, mock_load_data, mock_load_config, mock_load_env):
        """Test successful raw data retrieval"""
        # Setup
        mock_config = {
            "data_source": {
                "raw_path": "data/raw.csv",
                "type": "csv",
                "delimiter": ",",
                "header": 0,
                "encoding": "utf-8"
            }
        }
        mock_load_config.return_value = mock_config
        mock_df = pd.DataFrame({"col1": [1, 2]})
        mock_load_data.return_value = mock_df
        
        # Execute
        result = get_data(data_stage="raw")
        
        # Assert
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
    
    @patch('data_loader.load_env')
    @patch('data_loader.load_config')
    @patch('data_loader.load_data')
    def test_get_data_processed_success(self, mock_load_data, mock_load_config, mock_load_env):
        """Test successful processed data retrieval"""
        # Setup
        mock_config = {
            "data_source": {
                "processed_path": "data/processed.csv",
                "type": "csv"
            }
        }
        mock_load_config.return_value = mock_config
        mock_df = pd.DataFrame({"col1": [1, 2]})
        mock_load_data.return_value = mock_df
        
        # Execute
        result = get_data(data_stage="processed")
        
        # Assert
        mock_load_data.assert_called_once_with(
            path="data/processed.csv",
            file_type="csv",
            sheet_name=None,
            delimiter=",",
            header=0,
            encoding="utf-8"
        )
        pd.testing.assert_frame_equal(result, mock_df)
    
    @patch('data_loader.load_env')
    @patch('data_loader.load_config')
    @patch('data_loader.logger')
    def test_get_data_unknown_stage(self, mock_logger, mock_load_config, mock_load_env):
        """Test get_data with unknown data stage"""
        # Setup
        mock_config = {"data_source": {}}
        mock_load_config.return_value = mock_config
        
        # Execute & Assert
        with pytest.raises(ValueError, match="Unknown data_stage: invalid"):
            get_data(data_stage="invalid")
        
        mock_logger.error.assert_called_once_with("Unknown data_stage: %s", "invalid")
    
    @patch('data_loader.load_env')
    @patch('data_loader.load_config')
    @patch('data_loader.logger')
    def test_get_data_missing_raw_path(self, mock_logger, mock_load_config, mock_load_env):
        """Test get_data with missing raw path"""
        # Setup
        mock_config = {"data_source": {}}  # No raw_path
        mock_load_config.return_value = mock_config
        
        # Execute & Assert
        with pytest.raises(ValueError, match="Missing path for data_stage='raw'"):
            get_data(data_stage="raw")
        
        mock_logger.error.assert_called_once_with("No valid path for data_stage='%s'", "raw")
    
    @patch('data_loader.load_env')
    @patch('data_loader.load_config')
    @patch('data_loader.logger')
    def test_get_data_missing_processed_path(self, mock_logger, mock_load_config, mock_load_env):
        """Test get_data with missing processed path"""
        # Setup
        mock_config = {"data_source": {"raw_path": "data.csv"}}  # No processed_path
        mock_load_config.return_value = mock_config
        
        # Execute & Assert
        with pytest.raises(ValueError, match="Missing path for data_stage='processed'"):
            get_data(data_stage="processed")
        
        mock_logger.error.assert_called_once_with("No valid path for data_stage='%s'", "processed")
    
    @patch('data_loader.load_env')
    @patch('data_loader.load_config')
    @patch('data_loader.logger')
    def test_get_data_empty_path(self, mock_logger, mock_load_config, mock_load_env):
        """Test get_data with empty path"""
        # Setup
        mock_config = {"data_source": {"raw_path": ""}}  # Empty path
        mock_load_config.return_value = mock_config
        
        # Execute & Assert
        with pytest.raises(ValueError, match="Missing path for data_stage='raw'"):
            get_data(data_stage="raw")
        
        mock_logger.error.assert_called_once_with("No valid path for data_stage='%s'", "raw")
    
    @patch('data_loader.load_env')
    @patch('data_loader.load_config')
    @patch('data_loader.load_data')
    def test_get_data_excel_with_sheet(self, mock_load_data, mock_load_config, mock_load_env):
        """Test get_data with Excel file and sheet name"""
        # Setup
        mock_config = {
            "data_source": {
                "raw_path": "data.xlsx",
                "type": "excel",
                "sheet_name": "Data",
                "header": 1
            }
        }
        mock_load_config.return_value = mock_config
        mock_df = pd.DataFrame({"col1": [1, 2]})
        mock_load_data.return_value = mock_df
        
        # Execute
        result = get_data(data_stage="raw")
        
        # Assert
        mock_load_data.assert_called_once_with(
            path="data.xlsx",
            file_type="excel",
            sheet_name="Data",
            delimiter=",",
            header=1,
            encoding="utf-8"
        )
    
    def test_get_data_default_parameters(self):
        """Test get_data with default parameters"""
        with patch('data_loader.load_env') as mock_load_env:
            with patch('data_loader.load_config') as mock_load_config:
                mock_config = {"data_source": {"raw_path": "data.csv"}}
                mock_load_config.return_value = mock_config
                
                with patch('data_loader.load_data') as mock_load_data:
                    mock_load_data.return_value = pd.DataFrame()
                    
                    get_data()  # All defaults
                    
                    mock_load_env.assert_called_once_with(".env")
                    mock_load_config.assert_called_once_with("config.yaml")


class TestDataLoaderIntegration:
    """Integration tests for data loader functions"""
    
    def test_load_config_with_real_yaml(self, tmp_path: Path):
        """Test load_config with real YAML file"""
        # Create temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "data_source": {
                "raw_path": "data/raw.csv",
                "type": "csv",
                "header": 0
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Execute
        result = load_config(str(config_file))
        
        # Assert
        assert result == config_data
        assert result["data_source"]["raw_path"] == "data/raw.csv"
    
    def test_load_data_with_real_csv(self, tmp_path: Path):
        """Test load_data with real CSV file"""
        # Create temporary CSV file
        csv_file = tmp_path / "test_data.csv"
        test_data = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "target": [0, 1, 1]
        })
        test_data.to_csv(csv_file, index=False)
        
        # Execute
        result = load_data(str(csv_file), file_type="csv")
        
        # Assert
        pd.testing.assert_frame_equal(result, test_data)
        assert len(result) == 3
        assert list(result.columns) == ["feature1", "feature2", "target"]
    
    def test_load_data_with_real_excel(self, tmp_path: Path):
        """Test load_data with real Excel file"""
        # Create temporary Excel file
        excel_file = tmp_path / "test_data.xlsx"
        test_data = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "target": [0, 1, 1]
        })
        test_data.to_excel(excel_file, index=False)
        
        # Execute
        result = load_data(str(excel_file), file_type="excel")
        
        # Assert
        pd.testing.assert_frame_equal(result, test_data)
        assert len(result) == 3
        assert list(result.columns) == ["feature1", "feature2", "target"]


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @patch('data_loader.os.path.isfile')
    @patch('data_loader.pd.read_csv')
    @patch('data_loader.logger')
    def test_load_data_pandas_exception(self, mock_logger, mock_read_csv, mock_isfile):
        """Test load_data when pandas raises an exception"""
        # Setup
        mock_isfile.return_value = True
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No data")
        
        # Execute & Assert
        with pytest.raises(pd.errors.EmptyDataError):
            load_data("empty.csv")
        
        mock_logger.exception.assert_called_once_with("Data loading failed.")
    
    def test_get_data_missing_data_source_config(self):
        """Test get_data with completely missing data_source config"""
        with patch('data_loader.load_env'):
            with patch('data_loader.load_config') as mock_load_config:
                mock_load_config.return_value = {}  # No data_source key
                
                with pytest.raises(ValueError):
                    get_data(data_stage="raw")
    
    @patch('data_loader.load_env')
    @patch('data_loader.load_config')
    def test_get_data_config_load_error(self, mock_load_config, mock_load_env):
        """Test get_data when config loading fails"""
        # Setup
        mock_load_config.side_effect = FileNotFoundError("Config not found")
        
        # Execute & Assert
        with pytest.raises(FileNotFoundError):
            get_data()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])