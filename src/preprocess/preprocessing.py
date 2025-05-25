"""
preprocessing.py

Preprocess raw data using config-defined steps:
- Rename columns
- Drop irrelevant or sparse features
- Handle missing values
- Encode categorical features
- Scale numerical features

All steps are dynamically configured via config.yaml
"""

import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting preprocessing...")

    # 1. Rename columns
    renaming = config["preprocessing"].get("rename_columns", {})
    df.rename(columns=renaming, inplace=True)
    logger.info(f"Renamed columns: {list(renaming.values())}")

    # 2. Drop columns
    drop_cols = config["preprocessing"].get("drop_columns", [])
    df.drop(columns=drop_cols, errors='ignore', inplace=True)
    logger.info(f"Dropped columns: {drop_cols}")

    # 3. Handle missing values
    fillna_cfg = config["preprocessing"].get("fillna", {})

    for col in fillna_cfg.get("numerical_zero", []):
        df[col] = df[col].fillna(0)
    for col in fillna_cfg.get("categorical_none", []):
        df[col] = df[col].fillna("None")

    group_cfg = config["preprocessing"].get("fillna_groupby", {})
    if group_cfg:
        col = group_cfg["column"]
        grp = group_cfg["groupby"]
        df[col] = df.groupby(grp)[col].transform(lambda x: x.fillna(x.mean()))

    # 4. Drop rows with NA in critical columns
    dropna = config["preprocessing"].get("dropna_rows", [])
    df.dropna(subset=dropna, inplace=True)
    logger.info(f"Dropped rows with missing values in: {dropna}")

    # 5. Define transformers
    features = config["features"]
    categorical = features.get("categorical", [])
    ordinal = features.get("ordinal", [])
    continuous = features.get("continuous", [])

    transformers = []

    # One-hot encoding
    encoding_cfg = config["preprocessing"].get("encoding", {}).get("one_hot", {})
    if encoding_cfg:
        ohe = OneHotEncoder(handle_unknown=encoding_cfg.get("handle_unknown", "ignore"), drop=encoding_cfg.get("drop", None), sparse=False)
        transformers.append(("onehot", ohe, categorical))

    # Scaling
    scale_cfg = config["preprocessing"].get("scaling", {})
    if scale_cfg:
        scale_type = scale_cfg.get("method", "standard")
        if scale_type == "standard":
            scaler = StandardScaler()
        elif scale_type == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler: {scale_type}")
        numeric_cols = []
        if "continuous" in scale_cfg.get("apply_to", []):
            numeric_cols += continuous
        if "ordinal" in scale_cfg.get("apply_to", []):
            numeric_cols += ordinal
        transformers.append(("scaler", scaler, numeric_cols))

    # Column transformer
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    logger.info("Fitting transformers...")
    df_transformed = preprocessor.fit_transform(df)

    # Get transformed feature names
    try:
        ohe_features = preprocessor.named_transformers_["onehot"].get_feature_names_out(categorical)
    except:
        ohe_features = []

    final_columns = list(ohe_features) + [col for col in df.columns if col not in categorical]

    processed_df = pd.DataFrame(df_transformed, columns=final_columns)
    logger.info("Preprocessing completed.")

    return processed_df

if __name__ == "__main__":
    """
    Entry point for testing preprocessing script manually.
    Loads data using the configured path and runs preprocessing.
    """
    raw_path = config["data_source"]["raw_path"]
    df = pd.read_csv(raw_path)
    df_processed = preprocess_data(df)
    print("Preprocessing completed. Processed shape:", df_processed.shape)
    df_processed.head()
