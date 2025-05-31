"""
preprocessing.py

Preprocess raw data using config-defined steps:
- Rename columns
- Drop irrelevant or sparse features
- Handle missing values
- Encode categorical features
- Scale numerical features
"""

import pandas as pd
import pickle
import logging
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from typing import Tuple

logger = logging.getLogger(__name__)


def clean_raw_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Basic preprocessing steps applied before pipeline fitting or transforming."""
    logger.info("Starting raw data cleaning...")

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
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in fillna_cfg.get("categorical_none", []):
        if col in df.columns:
            df[col] = df[col].fillna("None")

    group_cfg = config["preprocessing"].get("fillna_groupby", {})
    if group_cfg:
        col = group_cfg["column"]
        grp = group_cfg["groupby"]
        if col in df.columns and grp in df.columns:
            df[col] = df.groupby(grp)[col].transform(
                lambda x: x.fillna(x.mean()))

    # 4. Drop rows with NA in critical columns
    dropna = config["preprocessing"].get("dropna_rows", [])
    df.dropna(
        subset=[col for col in dropna if col in df.columns], inplace=True)
    logger.info(f"Dropped rows with missing values in: {dropna}")

    return df


def build_preprocessing_pipeline(config: dict, df: pd.DataFrame) -> Tuple[ColumnTransformer, list]:
    """Construct the ColumnTransformer using the config."""
    features = config["features"]
    categorical = [col for col in features.get(
        "categorical", []) if col in df.columns]
    ordinal = [col for col in features.get("ordinal", []) if col in df.columns]
    continuous = [col for col in features.get(
        "continuous", []) if col in df.columns]

    transformers = []

    # One-hot encoding
    encoding_cfg = config["preprocessing"].get(
        "encoding", {}).get("one_hot", {})
    if encoding_cfg:
        ohe = OneHotEncoder(handle_unknown=encoding_cfg.get("handle_unknown", "ignore"),
                            drop=encoding_cfg.get("drop", None),
                            sparse_output=False)
        transformers.append(("onehot", ohe, categorical))

    # Scaling
    scale_cfg = config["preprocessing"].get("scaling", {})
    if scale_cfg:
        scale_type = scale_cfg.get("method", "standard")
        scaler = StandardScaler() if scale_type == "standard" else MinMaxScaler()
        numeric_cols = []
        if "continuous" in scale_cfg.get("apply_to", []):
            numeric_cols += continuous
        if "ordinal" in scale_cfg.get("apply_to", []):
            numeric_cols += ordinal
        transformers.append(("scaler", scaler, numeric_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop")
    selected_features = categorical + ordinal + continuous
    return preprocessor, selected_features


def fit_and_save_pipeline(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fit the preprocessing pipeline and save it to disk."""
    df_clean = clean_raw_data(df.copy(), config)
    pipeline, selected_features = build_preprocessing_pipeline(
        config, df_clean)

    logger.info("Fitting preprocessing pipeline...")
    df_transformed = pipeline.fit_transform(df_clean)

    pipeline_path = config["artifacts"]["preprocessing_pipeline"]
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Saved preprocessing pipeline to {pipeline_path}")

    # Save selected features
    features_path = config["artifacts"]["selected_features"]
    pd.Series(selected_features).to_json(features_path)
    logger.info(f"Saved selected features to {features_path}")

    try:
        feature_names = pipeline.get_feature_names_out()
    except AttributeError:
        feature_names = [f"f{i}" for i in range(df_transformed.shape[1])]

    return pd.DataFrame(df_transformed, columns=feature_names)


def transform_with_pipeline(df: pd.DataFrame, config: dict, pipeline: ColumnTransformer) -> pd.DataFrame:
    """Apply a previously fitted preprocessing pipeline."""
    df_clean = clean_raw_data(df.copy(), config)
    df_transformed = pipeline.transform(df_clean)

    try:
        feature_names = pipeline.get_feature_names_out()
    except AttributeError:
        feature_names = [f"f{i}" for i in range(df_transformed.shape[1])]

    return pd.DataFrame(df_transformed, columns=feature_names)
