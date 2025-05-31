import pandas as pd
from features import engineer_features

def test_engineer_features_creates_expected_columns():
    # Create a minimal mock dataframe
    df = pd.DataFrame({
        "1st Flr SF": [1000, 800],
        "2nd Flr SF": [500, 600],
        "Total Bsmt SF": [800, 700],
        "Full Bath": [2, 1],
        "Half Bath": [1, 2],
        "Bsmt Full Bath": [1, 0],
        "Bsmt Half Bath": [1, 1],
        "Yr Sold": [2020, 2020],
        "Year Built": [2000, 1995],
        "Year Remod/Add": [2010, 2005]
    })

    # Dummy config (not used currently)
    config = {}

    result = engineer_features(df, config)

    # Check that new columns exist
    for col in ["total_sf", "bathrooms", "house_age", "since_remodel"]:
        assert col in result.columns

def test_engineer_features_computes_correct_values():
    df = pd.DataFrame({
        "1st Flr SF": [1000],
        "2nd Flr SF": [500],
        "Total Bsmt SF": [800],
        "Full Bath": [2],
        "Half Bath": [1],
        "Bsmt Full Bath": [1],
        "Bsmt Half Bath": [1],
        "Yr Sold": [2020],
        "Year Built": [2000],
        "Year Remod/Add": [2010]
    })

    result = engineer_features(df, {})

    assert result["total_sf"].iloc[0] == 2300  # 1000 + 500 + 800
    assert result["bathrooms"].iloc[0] == 4.0  # 2 + 0.5 + 1 + 0.5
    assert result["house_age"].iloc[0] == 20   # 2020_]()
