import pandas as pd
import pytest
import numpy as np
import sys
import os

# Add the parent directory to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try different possible import paths based on your project structure
    from src.features.features import engineer_features
except ImportError:
    try:
        from src.features import engineer_features
    except ImportError:
        try:
            # If features.py is directly in src/
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
            from features import engineer_features
        except ImportError:
            try:
                # If features.py is in src/features/ directory
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'features'))
                from features import engineer_features
            except ImportError:
                print("Debug: Available files in src directory:")
                src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
                if os.path.exists(src_path):
                    print(os.listdir(src_path))
                    
                    # Check features subdirectory
                    features_path = os.path.join(src_path, 'features')
                    if os.path.exists(features_path):
                        print("Files in src/features directory:")
                        print(os.listdir(features_path))
                
                print("Current working directory:")
                print(os.getcwd())
                
                raise ImportError("Could not import engineer_features. Please check the module location.")

print(f"Successfully imported engineer_features")


class TestEngineerFeatures:
    """Test cases for engineer_features function"""
    
    def test_engineer_features_basic_calculations(self):
        """Test basic feature engineering calculations with valid data"""
        # Create sample data
        df = pd.DataFrame({
            "1st Flr SF": [1000, 1200, 800],
            "2nd Flr SF": [500, 0, 600],
            "Total Bsmt SF": [1000, 1200, 0],
            "Full Bath": [2, 1, 3],
            "Half Bath": [1, 0, 2],
            "Bsmt Full Bath": [1, 1, 0],
            "Bsmt Half Bath": [0, 1, 1],
            "Yr Sold": [2020, 2021, 2019],
            "Year Built": [2000, 1990, 2010],
            "Year Remod/Add": [2005, 1990, 2015]
        })
        
        config = {}  # Empty config for now
        
        result = engineer_features(df, config)
        
        # Test total_sf calculation
        expected_total_sf = [
            1000 + 500 + 1000,  # 2500
            1200 + 0 + 1200,    # 2400
            800 + 600 + 0       # 1400
        ]
        assert result["total_sf"].tolist() == expected_total_sf
        
        # Test bathrooms calculation
        expected_bathrooms = [
            2 + 0.5 * 1 + 1 + 0.5 * 0,  # 3.5
            1 + 0.5 * 0 + 1 + 0.5 * 1,  # 2.5
            3 + 0.5 * 2 + 0 + 0.5 * 1   # 4.5
        ]
        assert result["bathrooms"].tolist() == expected_bathrooms
        
        # Test house_age calculation
        expected_house_age = [
            2020 - 2000,  # 20
            2021 - 1990,  # 31
            2019 - 2010   # 9
        ]
        assert result["house_age"].tolist() == expected_house_age
        
        # Test since_remodel calculation
        expected_since_remodel = [
            2020 - 2005,  # 15
            2021 - 1990,  # 31
            2019 - 2015   # 4
        ]
        assert result["since_remodel"].tolist() == expected_since_remodel
    
    def test_engineer_features_returns_copy(self):
        """Test that engineer_features returns a copy and doesn't modify original"""
        df = pd.DataFrame({
            "1st Flr SF": [1000],
            "2nd Flr SF": [500],
            "Total Bsmt SF": [1000],
            "Full Bath": [2],
            "Half Bath": [1],
            "Bsmt Full Bath": [1],
            "Bsmt Half Bath": [0],
            "Yr Sold": [2020],
            "Year Built": [2000],
            "Year Remod/Add": [2005]
        })
        
        original_columns = df.columns.tolist()
        original_shape = df.shape
        
        config = {}
        result = engineer_features(df, config)
        
        # Original DataFrame should be unchanged
        assert df.columns.tolist() == original_columns
        assert df.shape == original_shape
        assert "total_sf" not in df.columns
        assert "bathrooms" not in df.columns
        assert "house_age" not in df.columns
        assert "since_remodel" not in df.columns
        
        # Result should have new columns
        assert "total_sf" in result.columns
        assert "bathrooms" in result.columns
        assert "house_age" in result.columns
        assert "since_remodel" in result.columns
    
    def test_engineer_features_with_zero_values(self):
        """Test feature engineering with zero values"""
        df = pd.DataFrame({
            "1st Flr SF": [1000, 0],
            "2nd Flr SF": [0, 0],
            "Total Bsmt SF": [0, 1000],
            "Full Bath": [0, 2],
            "Half Bath": [0, 0],
            "Bsmt Full Bath": [0, 0],
            "Bsmt Half Bath": [0, 0],
            "Yr Sold": [2020, 2021],
            "Year Built": [2020, 2000],  # Same year built as sold
            "Year Remod/Add": [2020, 2021]  # Remodel after built
        })
        
        config = {}
        result = engineer_features(df, config)
        
        # Test with zeros
        assert result["total_sf"].iloc[0] == 1000  # 1000 + 0 + 0
        assert result["total_sf"].iloc[1] == 1000  # 0 + 0 + 1000
        
        assert result["bathrooms"].iloc[0] == 0.0  # All bathroom counts are 0
        assert result["bathrooms"].iloc[1] == 2.0  # Only full bath = 2
        
        assert result["house_age"].iloc[0] == 0    # Built same year as sold
        assert result["house_age"].iloc[1] == 21   # 2021 - 2000
        
        assert result["since_remodel"].iloc[0] == 0  # Remodeled same year as sold
        assert result["since_remodel"].iloc[1] == 0  # 2021 - 2021
    
    def test_engineer_features_with_nan_values(self):
        """Test feature engineering behavior with NaN values"""
        df = pd.DataFrame({
            "1st Flr SF": [1000, np.nan, 800],
            "2nd Flr SF": [500, 600, np.nan],
            "Total Bsmt SF": [1000, 1200, 0],
            "Full Bath": [2, np.nan, 3],
            "Half Bath": [1, 0, np.nan],
            "Bsmt Full Bath": [1, 1, 0],
            "Bsmt Half Bath": [0, 1, 1],
            "Yr Sold": [2020, 2021, 2019],
            "Year Built": [2000, np.nan, 2010],
            "Year Remod/Add": [2005, 1990, np.nan]
        })
        
        config = {}
        result = engineer_features(df, config)
        
        # Check that NaN propagates correctly in calculations
        assert pd.isna(result["total_sf"].iloc[1])  # nan + 600 + 1200 = nan
        assert pd.isna(result["total_sf"].iloc[2])  # 800 + nan + 0 = nan
        
        assert pd.isna(result["bathrooms"].iloc[1])  # Contains nan
        assert pd.isna(result["bathrooms"].iloc[2])  # Contains nan
        
        assert pd.isna(result["house_age"].iloc[1])  # 2021 - nan = nan
        assert pd.isna(result["since_remodel"].iloc[2])  # 2019 - nan = nan
    
    def test_engineer_features_with_negative_values(self):
        """Test feature engineering with edge case negative results"""
        df = pd.DataFrame({
            "1st Flr SF": [1000],
            "2nd Flr SF": [500],
            "Total Bsmt SF": [1000],
            "Full Bath": [2],
            "Half Bath": [1],
            "Bsmt Full Bath": [1],
            "Bsmt Half Bath": [0],
            "Yr Sold": [1990],     # Sold before built (edge case)
            "Year Built": [2000],
            "Year Remod/Add": [2005]  # Remodeled after sold
        })
        
        config = {}
        result = engineer_features(df, config)
        
        # Should handle negative ages (though unrealistic)
        assert result["house_age"].iloc[0] == -10   # 1990 - 2000
        assert result["since_remodel"].iloc[0] == -15  # 1990 - 2005
    
    def test_engineer_features_preserves_original_columns(self):
        """Test that all original columns are preserved"""
        df = pd.DataFrame({
            "1st Flr SF": [1000, 1200],
            "2nd Flr SF": [500, 0],
            "Total Bsmt SF": [1000, 1200],
            "Full Bath": [2, 1],
            "Half Bath": [1, 0],
            "Bsmt Full Bath": [1, 1],
            "Bsmt Half Bath": [0, 1],
            "Yr Sold": [2020, 2021],
            "Year Built": [2000, 1990],
            "Year Remod/Add": [2005, 1990],
            "Other Column": ["A", "B"],  # Additional column that should be preserved
            "Id": [1, 2]
        })
        
        config = {}
        result = engineer_features(df, config)
        
        # All original columns should still be there
        for col in df.columns:
            assert col in result.columns
            pd.testing.assert_series_equal(df[col], result[col])
        
        # New columns should be added
        new_columns = ["total_sf", "bathrooms", "house_age", "since_remodel"]
        for col in new_columns:
            assert col in result.columns
        
        # Should have original columns + 4 new columns
        assert len(result.columns) == len(df.columns) + 4
    
    def test_engineer_features_with_large_values(self):
        """Test feature engineering with large numeric values"""
        df = pd.DataFrame({
            "1st Flr SF": [5000],
            "2nd Flr SF": [3000],
            "Total Bsmt SF": [2000],
            "Full Bath": [10],
            "Half Bath": [5],
            "Bsmt Full Bath": [3],
            "Bsmt Half Bath": [2],
            "Yr Sold": [2023],
            "Year Built": [1850],  # Very old house
            "Year Remod/Add": [1900]
        })
        
        config = {}
        result = engineer_features(df, config)
        
        assert result["total_sf"].iloc[0] == 10000  # 5000 + 3000 + 2000
        assert result["bathrooms"].iloc[0] == 16.5  # 10 + 2.5 + 3 + 1
        assert result["house_age"].iloc[0] == 173   # 2023 - 1850
        assert result["since_remodel"].iloc[0] == 123  # 2023 - 1900
    
    def test_engineer_features_data_types(self):
        """Test that engineered features have appropriate data types"""
        df = pd.DataFrame({
            "1st Flr SF": [1000],
            "2nd Flr SF": [500],
            "Total Bsmt SF": [1000],
            "Full Bath": [2],
            "Half Bath": [1],
            "Bsmt Full Bath": [1],
            "Bsmt Half Bath": [0],
            "Yr Sold": [2020],
            "Year Built": [2000],
            "Year Remod/Add": [2005]
        })
        
        config = {}
        result = engineer_features(df, config)
        
        # Check data types of engineered features
        assert pd.api.types.is_numeric_dtype(result["total_sf"])
        assert pd.api.types.is_numeric_dtype(result["bathrooms"])
        assert pd.api.types.is_numeric_dtype(result["house_age"])
        assert pd.api.types.is_numeric_dtype(result["since_remodel"])
    
    def test_engineer_features_empty_dataframe(self):
        """Test behavior with empty DataFrame"""
        df = pd.DataFrame()
        config = {}
        
        # Should handle empty DataFrame gracefully
        with pytest.raises((KeyError, AttributeError)):
            engineer_features(df, config)
    
    def test_engineer_features_missing_required_columns(self):
        """Test behavior when required columns are missing"""
        df = pd.DataFrame({
            "1st Flr SF": [1000],
            "2nd Flr SF": [500],
            # Missing other required columns
        })
        config = {}
        
        # Should raise KeyError for missing columns
        with pytest.raises(KeyError):
            engineer_features(df, config)
    
    def test_engineer_features_config_parameter(self):
        """Test that config parameter is accepted (even if not used)"""
        df = pd.DataFrame({
            "1st Flr SF": [1000],
            "2nd Flr SF": [500],
            "Total Bsmt SF": [1000],
            "Full Bath": [2],
            "Half Bath": [1],
            "Bsmt Full Bath": [1],
            "Bsmt Half Bath": [0],
            "Yr Sold": [2020],
            "Year Built": [2000],
            "Year Remod/Add": [2005]
        })
        
        # Test with different config values
        config1 = {}
        config2 = {"some_setting": "value"}
        config3 = {"feature_engineering": {"enabled": True}}
        
        # All should work the same way (config not currently used)
        result1 = engineer_features(df, config1)
        result2 = engineer_features(df, config2)
        result3 = engineer_features(df, config3)
        
        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result2, result3)


# Fixtures for common test data
@pytest.fixture
def sample_house_data():
    """Sample house data for testing"""
    return pd.DataFrame({
        "1st Flr SF": [1000, 1200, 800, 1500],
        "2nd Flr SF": [500, 0, 600, 700],
        "Total Bsmt SF": [1000, 1200, 0, 1000],
        "Full Bath": [2, 1, 3, 2],
        "Half Bath": [1, 0, 2, 1],
        "Bsmt Full Bath": [1, 1, 0, 1],
        "Bsmt Half Bath": [0, 1, 1, 0],
        "Yr Sold": [2020, 2021, 2019, 2022],
        "Year Built": [2000, 1990, 2010, 1985],
        "Year Remod/Add": [2005, 1990, 2015, 2000],
        "Id": [1, 2, 3, 4]
    })


@pytest.fixture
def empty_config():
    """Empty configuration dictionary"""
    return {}


def test_engineer_features_with_fixtures(sample_house_data, empty_config):
    """Test using fixtures"""
    result = engineer_features(sample_house_data, empty_config)
    
    # Should have all original columns plus 4 new ones
    assert len(result.columns) == len(sample_house_data.columns) + 4
    
    # Check that calculations work for all rows
    assert len(result) == len(sample_house_data)
    assert result["total_sf"].notna().all()
    assert result["bathrooms"].notna().all()
    assert result["house_age"].notna().all()
    assert result["since_remodel"].notna().all()


# Test runner configuration
if __name__ == "__main__":
    # Configure for verbose testing
    pytest.main(["-v", __file__])