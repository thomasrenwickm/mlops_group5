import os
import pytest
import pandas as pd
import json
import tempfile
import yaml
import logging
import sys
from unittest.mock import patch, mock_open, MagicMock, call
from io import StringIO

# Add the parent directory to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the config.yaml loading during import
mock_config_data = {
    "data_validation": {
        "schema": {
            "columns": [
                {"name": "id", "dtype": "int", "required": True},
                {"name": "name", "dtype": "str", "required": True}
            ]
        },
        "action_on_error": "raise",
        "report_path": "logs/validation_report.json"
    },
    "data_source": {
        "raw_path": "data/raw.csv"
    }
}

mock_yaml_content = """
data_validation:
  schema:
    columns:
      - name: id
        dtype: int
        required: true
      - name: name
        dtype: str
        required: true
  action_on_error: raise
  report_path: logs/validation_report.json
data_source:
  raw_path: data/raw.csv
"""

# Mock the file operations during import to prevent config.yaml loading issues
with patch('builtins.open', mock_open(read_data=mock_yaml_content)):
    with patch('yaml.safe_load', return_value=mock_config_data):
        try:
            # Try different possible import paths based on your project structure
            from src.data_validation.data_validation import validate_schema
            MODULE_PATH = 'src.data_validation.data_validation'
        except ImportError:
            try:
                from src.data_validation import validate_schema
                MODULE_PATH = 'src.data_validation'
            except ImportError:
                try:
                    # If data_validation.py is directly in src/
                    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
                    from data_validation import validate_schema
                    MODULE_PATH = 'data_validation'
                except ImportError:
                    # If data_validation.py is in the tests directory or current directory
                    try:
                        from data_validation import validate_schema
                        MODULE_PATH = 'data_validation'
                    except ImportError:
                        print("Available files in src directory:")
                        src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
                        if os.path.exists(src_path):
                            print(os.listdir(src_path))
                        else:
                            print("src directory not found")
                        
                        print("Current working directory:")
                        print(os.getcwd())
                        print("Files in current directory:")
                        print(os.listdir('.'))
                        
                        raise ImportError("Could not import validate_schema. Please check the module location.")

print(f"Successfully imported validate_schema from {MODULE_PATH}")


class TestValidateSchema:
    """Test cases for validate_schema function"""
    
    def test_validate_schema_success_all_columns_valid(self):
        """Test successful validation with all required columns and correct types"""
        # Create test DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [85.5, 92.0, 78.3],
            'active': [True, False, True]
        })
        
        # Define schema
        schema = [
            {"name": "id", "dtype": "int", "required": True},
            {"name": "name", "dtype": "str", "required": True},
            {"name": "score", "dtype": "float", "required": True},
            {"name": "active", "dtype": "bool", "required": False}
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()) as mock_file:
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    result = validate_schema(df, schema, action="raise")
        
        # Assertions
        assert result["status"] == "pass"
        assert result["error_count"] == 0
        assert result["errors"] == []
        mock_logger.info.assert_called_once_with("Data validation passed with no errors.")
    
    def test_validate_schema_missing_required_column(self):
        """Test validation failure when required column is missing"""
        # Create test DataFrame without 'email' column
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        # Define schema with required email column
        schema = [
            {"name": "id", "dtype": "int", "required": True},
            {"name": "name", "dtype": "str", "required": True},
            {"name": "email", "dtype": "str", "required": True}  # Missing column
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()) as mock_file:
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    with pytest.raises(ValueError, match="Data validation failed with 1 error"):
                        validate_schema(df, schema, action="raise")
        
        # Check that error was logged
        mock_logger.error.assert_called_once_with("Missing required column: email")
    
    def test_validate_schema_missing_optional_column(self):
        """Test validation success when optional column is missing"""
        # Create test DataFrame without 'phone' column
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        # Define schema with optional phone column
        schema = [
            {"name": "id", "dtype": "int", "required": True},
            {"name": "name", "dtype": "str", "required": True},
            {"name": "phone", "dtype": "str", "required": False}  # Optional missing column
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()) as mock_file:
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    result = validate_schema(df, schema, action="raise")
        
        # Should pass because phone is optional
        assert result["status"] == "pass"
        assert result["error_count"] == 0
        mock_logger.info.assert_called_once_with("Data validation passed with no errors.")
    
    def test_validate_schema_wrong_int_type(self):
        """Test validation failure when column has wrong integer type"""
        # Create DataFrame with string instead of int
        df = pd.DataFrame({
            'id': ['1', '2', '3'],  # String instead of int
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        schema = [
            {"name": "id", "dtype": "int", "required": True},
            {"name": "name", "dtype": "str", "required": True}
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()) as mock_file:
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    with pytest.raises(ValueError, match="Data validation failed with 1 error"):
                        validate_schema(df, schema, action="raise")
        
        # Check error message
        expected_error = "Column 'id' expected int but found object"
        mock_logger.error.assert_called_once_with(expected_error)
    
    def test_validate_schema_wrong_float_type(self):
        """Test validation failure when column has wrong float type"""
        # Create DataFrame with string instead of float
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'score': ['85.5', '92.0', '78.3']  # String instead of float
        })
        
        schema = [
            {"name": "id", "dtype": "int", "required": True},
            {"name": "score", "dtype": "float", "required": True}
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()) as mock_file:
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    with pytest.raises(ValueError, match="Data validation failed with 1 error"):
                        validate_schema(df, schema, action="raise")
        
        # Check error message
        expected_error = "Column 'score' expected float but found object"
        mock_logger.error.assert_called_once_with(expected_error)
    
    def test_validate_schema_wrong_string_type(self):
        """Test validation failure when column has wrong string type"""
        # Create DataFrame with numeric instead of string
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': [123, 456, 789]  # Numeric instead of string
        })
        
        schema = [
            {"name": "id", "dtype": "int", "required": True},
            {"name": "name", "dtype": "str", "required": True}
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()) as mock_file:
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    with pytest.raises(ValueError, match="Data validation failed with 1 error"):
                        validate_schema(df, schema, action="raise")
        
        # Check error message contains expected information
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        assert "Column 'name' expected str but found" in error_call
    
    def test_validate_schema_multiple_errors(self):
        """Test validation with multiple errors"""
        # Create DataFrame with multiple issues
        df = pd.DataFrame({
            'id': ['1', '2', '3'],  # Wrong type
            'score': ['85.5', '92.0', '78.3']  # Wrong type
            # Missing required 'name' column
        })
        
        schema = [
            {"name": "id", "dtype": "int", "required": True},
            {"name": "name", "dtype": "str", "required": True},  # Missing
            {"name": "score", "dtype": "float", "required": True}
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()) as mock_file:
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    with pytest.raises(ValueError, match="Data validation failed with 3 error"):
                        validate_schema(df, schema, action="raise")
        
        # Should have 3 error calls: missing column, wrong id type, wrong score type
        assert mock_logger.error.call_count == 3
    
    def test_validate_schema_warn_action(self):
        """Test validation with 'warn' action instead of 'raise'"""
        # Create DataFrame with error
        df = pd.DataFrame({
            'id': ['1', '2', '3']  # Wrong type
        })
        
        schema = [
            {"name": "id", "dtype": "int", "required": True}
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()) as mock_file:
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    result = validate_schema(df, schema, action="warn")
        
        # Should not raise exception, but should warn
        assert result["status"] == "fail"
        assert result["error_count"] == 1
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Validation completed with warnings" in warning_call
    
    def test_validate_schema_report_generation(self):
        """Test that validation report is written correctly"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        schema = [
            {"name": "id", "dtype": "int", "required": True},
            {"name": "name", "dtype": "str", "required": True}
        ]
        
        mock_file_handle = mock_open()
        
        with patch(f'{MODULE_PATH}.os.makedirs') as mock_makedirs:
            with patch(f'{MODULE_PATH}.open', mock_file_handle) as mock_file:
                with patch(f'{MODULE_PATH}.json.dump') as mock_json_dump:
                    with patch(f'{MODULE_PATH}.logger'):
                        result = validate_schema(df, schema, action="raise")
        
        # Check that directories were created
        mock_makedirs.assert_called_once()
        
        # Check that file was opened for writing
        mock_file.assert_called_once()
        
        # Check that JSON was written with correct structure
        mock_json_dump.assert_called_once()
        written_report = mock_json_dump.call_args[0][0]
        assert written_report["status"] == "pass"
        assert written_report["error_count"] == 0
        assert written_report["errors"] == []
    
    def test_validate_schema_report_path_creation(self):
        """Test that report directory is created if it doesn't exist"""
        df = pd.DataFrame({'id': [1, 2, 3]})
        schema = [{"name": "id", "dtype": "int", "required": True}]
        
        with patch(f'{MODULE_PATH}.os.makedirs') as mock_makedirs:
            with patch(f'{MODULE_PATH}.open', mock_open()):
                with patch(f'{MODULE_PATH}.logger'):
                    with patch(f'{MODULE_PATH}.report_path', 'logs/validation_report.json'):
                        validate_schema(df, schema, action="raise")
        
        # Check that makedirs was called with the directory
        mock_makedirs.assert_called_once_with('logs', exist_ok=True)
    
    def test_validate_schema_empty_dataframe(self):
        """Test validation with empty DataFrame"""
        df = pd.DataFrame()
        
        schema = [
            {"name": "id", "dtype": "int", "required": True}
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()):
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    with pytest.raises(ValueError, match="Data validation failed with 1 error"):
                        validate_schema(df, schema, action="raise")
        
        mock_logger.error.assert_called_once_with("Missing required column: id")
    
    def test_validate_schema_no_required_field_defaults_false(self):
        """Test that columns without 'required' field default to False"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        # Schema without 'required' field for optional_col
        schema = [
            {"name": "id", "dtype": "int", "required": True},
            {"name": "name", "dtype": "str", "required": True},
            {"name": "optional_col", "dtype": "str"}  # No 'required' field
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()):
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    result = validate_schema(df, schema, action="raise")
        
        # Should pass because optional_col defaults to not required
        assert result["status"] == "pass"
        assert result["error_count"] == 0
        mock_logger.info.assert_called_once_with("Data validation passed with no errors.")


class TestDataValidationIntegration:
    """Integration tests for the data validation module"""
    
    def test_integration_with_real_dataframe(self):
        """Test end-to-end validation with real DataFrame and temporary files"""
        # Create test DataFrame
        df = pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'username': ['alice', 'bob', 'charlie', 'diana'],
            'score': [85.5, 92.0, 78.3, 96.7],
            'active': [True, False, True, True]
        })
        
        # Define realistic schema
        schema = [
            {"name": "user_id", "dtype": "int", "required": True},
            {"name": "username", "dtype": "str", "required": True},
            {"name": "score", "dtype": "float", "required": True},
            {"name": "active", "dtype": "bool", "required": False},
            {"name": "optional_field", "dtype": "str", "required": False}
        ]
        
        # Create temporary directory for report
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "validation_report.json")
            
            with patch(f'{MODULE_PATH}.report_path', report_path):
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    result = validate_schema(df, schema, action="raise")
            
            # Verify result
            assert result["status"] == "pass"
            assert result["error_count"] == 0
            assert result["errors"] == []
            
            # Verify report file was created
            assert os.path.exists(report_path)
            
            # Verify report contents
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            assert report_data["status"] == "pass"
            assert report_data["error_count"] == 0
            assert report_data["errors"] == []
            
            # Verify logging
            mock_logger.info.assert_called_once_with("Data validation passed with no errors.")


class TestDataValidationEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_validate_schema_with_nan_values(self):
        """Test validation behavior with NaN values"""
        df = pd.DataFrame({
            'id': [1, 2, None],  # Contains NaN
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        schema = [
            {"name": "id", "dtype": "int", "required": True},
            {"name": "name", "dtype": "str", "required": True}
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()):
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    # Should handle NaN gracefully - behavior depends on pandas version
                    result = validate_schema(df, schema, action="warn")
                    
                    # At minimum, should not crash
                    assert "status" in result
                    assert "error_count" in result
                    assert "errors" in result
    
    def test_validate_schema_with_mixed_types(self):
        """Test validation with DataFrames containing mixed types"""
        df = pd.DataFrame({
            'mixed_col': [1, '2', 3.0, True],  # Mixed types
            'id': [1, 2, 3, 4]
        })
        
        schema = [
            {"name": "mixed_col", "dtype": "int", "required": True},
            {"name": "id", "dtype": "int", "required": True}
        ]
        
        with patch(f'{MODULE_PATH}.os.makedirs'):
            with patch(f'{MODULE_PATH}.open', mock_open()):
                with patch(f'{MODULE_PATH}.logger') as mock_logger:
                    result = validate_schema(df, schema, action="warn")
                    
                    # Should handle mixed types and provide meaningful validation
                    assert isinstance(result, dict)
                    assert "status" in result


# Fixtures for common test data
@pytest.fixture
def sample_valid_dataframe():
    """Sample valid DataFrame for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'score': [85.5, 92.0, 78.3, 96.7, 89.2],
        'active': [True, False, True, True, False]
    })


@pytest.fixture
def sample_valid_schema():
    """Sample valid schema for testing"""
    return [
        {"name": "id", "dtype": "int", "required": True},
        {"name": "name", "dtype": "str", "required": True},
        {"name": "score", "dtype": "float", "required": True},
        {"name": "active", "dtype": "bool", "required": False}
    ]


@pytest.fixture
def sample_invalid_dataframe():
    """Sample invalid DataFrame for testing"""
    return pd.DataFrame({
        'id': ['1', '2', '3'],  # Wrong type (string instead of int)
        'score': ['85.5', '92.0', '78.3']  # Wrong type (string instead of float)
        # Missing required 'name' column
    })


# Test runner configuration
if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Run tests with pytest
    pytest.main(["-v", __file__])