import pandas as pd
import pytest
from data_validation import validate_schema

# Sample schema (matches expected config format)
SAMPLE_SCHEMA = [
    {"name": "Id", "dtype": "int", "required": True},
    {"name": "SalePrice", "dtype": "float", "required": True},
    {"name": "Neighborhood", "dtype": "str", "required": True}
]

# Valid DataFrame that should pass
VALID_DF = pd.DataFrame({
    "Id": [1, 2, 3],
    "SalePrice": [200000.0, 250000.5, 180000.0],
    "Neighborhood": ["NAmes", "CollgCr", "OldTown"]
})

# Missing column
MISSING_COLUMN_DF = pd.DataFrame({
    "Id": [1, 2, 3],
    "SalePrice": [200000.0, 250000.5, 180000.0]
})

# Wrong dtype
WRONG_DTYPE_DF = pd.DataFrame({
    "Id": ["1", "2", "3"],
    "SalePrice": [200000.0, 250000.5, 180000.0],
    "Neighborhood": ["NAmes", "CollgCr", "OldTown"]
})


def test_validate_schema_passes_on_valid_data(tmp_path):
    result = validate_schema(VALID_DF, SAMPLE_SCHEMA, action="raise")
    assert result["status"] == "pass"
    assert result["error_count"] == 0

def test_validate_schema_fails_on_missing_column(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        validate_schema(MISSING_COLUMN_DF, SAMPLE_SCHEMA, action="raise")
    assert "Missing required column" in str(excinfo.value)

def test_validate_schema_fails_on_wrong_dtype(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        validate_schema(WRONG_DTYPE_DF, SAMPLE_SCHEMA, action="raise")
    assert "expected int but found" in str(excinfo.value)

def test_validate_schema_warns_instead_of_failing(caplog):
    result = validate_schema(WRONG_DTYPE_DF, SAMPLE_SCHEMA, action="warn")
    assert result["status"] == "fail"
    assert "expected int but found" in result["errors"][0]
