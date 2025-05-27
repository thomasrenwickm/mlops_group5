# run_model_test.py

import pandas as pd
import yaml
from src.model.model import run_model_pipeline  # adjust path if needed

# Load a sample of the data
df = pd.read_csv("dataset/raw/AMES_Housing_Data.csv").head(100)

# Load the config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Run the function
print("✅ Running model pipeline on sample data...")
run_model_pipeline(df, config)
print("✅ Model pipeline finished.")
