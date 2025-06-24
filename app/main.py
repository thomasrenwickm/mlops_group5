from __future__ import annotations

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import pickle
import json
import yaml

from src.preprocess.preprocessing import clean_raw_data, transform_with_pipeline
from src.features.features import engineer_features

# === Load Config and Artifacts ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

PIPELINE_PATH = PROJECT_ROOT / CONFIG["artifacts"]["preprocessing_pipeline"]
MODEL_PATH = PROJECT_ROOT / CONFIG["artifacts"]["model"]
FEATURES_PATH = PROJECT_ROOT / CONFIG["artifacts"]["selected_features"]

with PIPELINE_PATH.open("rb") as f:
    PIPELINE = pickle.load(f)

with MODEL_PATH.open("rb") as f:
    MODEL = pickle.load(f)

# with FEATURES_PATH.open("r") as f:
#     SELECTED_FEATURES = json.load(f)
with FEATURES_PATH.open("r") as f:
    raw = json.load(f)
    SELECTED_FEATURES = list(raw.values()) if isinstance(raw, dict) else raw


# === FastAPI App ===
app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# === Define Minimal Required Input Schema ===
# These are the MINIMAL fields needed to engineer features


class PredictionInput(BaseModel):

    overall_qual: int = Field(..., alias="Overall Qual")
    gr_liv_area: float = Field(..., alias="Gr Liv Area")
    garage_cars: int = Field(..., alias="Garage Cars")
    kitchen_qual: str = Field(..., alias="Kitchen Qual")
    exter_qual: str = Field(..., alias="Exter Qual")
    neighborhood: str = Field(..., alias="Neighborhood")
    bsmt_full_bath: int = Field(..., alias="Bsmt Full Bath")
    bsmt_half_bath: int = Field(..., alias="Bsmt Half Bath")
    full_bath: int = Field(..., alias="Full Bath")
    half_bath: int = Field(..., alias="Half Bath")
    first_flr_sf: int = Field(..., alias="1st Flr SF")
    second_flr_sf: int = Field(..., alias="2nd Flr SF")
    total_bsmt_sf: float = Field(..., alias="Total Bsmt SF")
    year_built: int = Field(..., alias="Year Built")
    year_remod_add: int = Field(..., alias="Year Remod/Add")
    yr_sold: int = Field(..., alias="Yr Sold")

    # Optional features for SelectKBest/Model inference
    ms_zoning: Optional[str] = Field(None, alias="MS Zoning")
    street: Optional[str] = Field(None, alias="Street")
    lot_shape: Optional[str] = Field(None, alias="Lot Shape")
    land_contour: Optional[str] = Field(None, alias="Land Contour")
    utilities: Optional[str] = Field(None, alias="Utilities")
    lot_config: Optional[str] = Field(None, alias="Lot Config")
    land_slope: Optional[str] = Field(None, alias="Land Slope")
    condition_1: Optional[str] = Field(None, alias="Condition 1")
    condition_2: Optional[str] = Field(None, alias="Condition 2")
    bldg_type: Optional[str] = Field(None, alias="Bldg Type")
    house_style: Optional[str] = Field(None, alias="House Style")
    overall_cond: Optional[int] = Field(None, alias="Overall Cond")
    roof_style: Optional[str] = Field(None, alias="Roof Style")
    roof_matl: Optional[str] = Field(None, alias="Roof Matl")
    exterior_1st: Optional[str] = Field(None, alias="Exterior 1st")
    exterior_2nd: Optional[str] = Field(None, alias="Exterior 2nd")
    mas_vnr_type: Optional[str] = Field(None, alias="Mas Vnr Type")
    mas_vnr_area: Optional[float] = Field(None, alias="Mas Vnr Area")
    exter_cond: Optional[str] = Field(None, alias="Exter Cond")
    foundation: Optional[str] = Field(None, alias="Foundation")
    bsmt_qual: Optional[str] = Field(None, alias="Bsmt Qual")
    bsmt_cond: Optional[str] = Field(None, alias="Bsmt Cond")
    bsmt_exposure: Optional[str] = Field(None, alias="Bsmt Exposure")
    bsmtfin_type_1: Optional[str] = Field(None, alias="BsmtFin Type 1")
    bsmtfin_sf_1: Optional[float] = Field(None, alias="BsmtFin SF 1")
    bsmtfin_type_2: Optional[str] = Field(None, alias="BsmtFin Type 2")
    bsmtfin_sf_2: Optional[float] = Field(None, alias="BsmtFin SF 2")
    bsmt_unf_sf: Optional[float] = Field(None, alias="Bsmt Unf SF")
    heating: Optional[str] = Field(None, alias="Heating")
    heating_qc: Optional[str] = Field(None, alias="Heating QC")
    central_air: Optional[str] = Field(None, alias="Central Air")
    electrical: Optional[str] = Field(None, alias="Electrical")
    kitchen_abvgr: Optional[int] = Field(None, alias="Kitchen AbvGr")
    totrms_abvgr: Optional[int] = Field(None, alias="TotRms AbvGr")
    functional: Optional[str] = Field(None, alias="Functional")
    fireplaces: Optional[int] = Field(None, alias="Fireplaces")
    fireplace_qu: Optional[str] = Field(None, alias="Fireplace Qu")
    garage_type: Optional[str] = Field(None, alias="Garage Type")
    garage_yr_blt: Optional[float] = Field(None, alias="Garage Yr Blt")
    garage_finish: Optional[str] = Field(None, alias="Garage Finish")
    garage_area: Optional[float] = Field(None, alias="Garage Area")
    garage_qual: Optional[str] = Field(None, alias="Garage Qual")
    garage_cond: Optional[str] = Field(None, alias="Garage Cond")
    paved_drive: Optional[str] = Field(None, alias="Paved Drive")
    wood_deck_sf: Optional[float] = Field(None, alias="Wood Deck SF")
    open_porch_sf: Optional[float] = Field(None, alias="Open Porch SF")
    enclosed_porch: Optional[float] = Field(None, alias="Enclosed Porch")
    three_ssn_porch: Optional[float] = Field(None, alias="3Ssn Porch")
    screen_porch: Optional[float] = Field(None, alias="Screen Porch")
    pool_area: Optional[float] = Field(None, alias="Pool Area")
    sale_type: Optional[str] = Field(None, alias="Sale Type")
    sale_condition: Optional[str] = Field(None, alias="Sale Condition")
    bedroom_abvgr: Optional[int] = Field(None, alias="Bedroom AbvGr")
    lot_area: Optional[int] = Field(None, alias="Lot Area")
    low_qual_fin_sf: Optional[int] = Field(None, alias="Low Qual Fin SF")

    class Config:
        # extra = "forbid"
        validate_by_name = True
        json_schema_extra = {
            "example": {

                "PID": 526350040,
                "MS SubClass": 20,
                "MS Zoning": "RH",
                "Lot Frontage": 80.0,
                "Lot Area": 11622,
                "Street": "Pave",
                "Alley": 'NA',
                "Lot Shape": "Reg",
                "Land Contour": "Lvl",
                "Utilities": "AllPub",
                "Lot Config": "Inside",
                "Land Slope": "Gtl",
                "Neighborhood": "NAmes",
                "Condition 1": "Feedr",
                "Condition 2": "Norm",
                "Bldg Type": "1Fam",
                "House Style": "1Story",
                "Overall Qual": 5,
                "Overall Cond": 6,
                "Year Built": 1961,
                "Year Remod/Add": 1961,
                "Roof Style": "Gable",
                "Roof Matl": "CompShg",
                "Exterior 1st": "VinylSd",
                "Exterior 2nd": "VinylSd",
                "Mas Vnr Type": "None",
                "Mas Vnr Area": 0.0,
                "Exter Qual": "TA",
                "Exter Cond": "TA",
                "Foundation": "CBlock",
                "Bsmt Qual": "TA",
                "Bsmt Cond": "TA",
                "Bsmt Exposure": "No",
                "BsmtFin Type 1": "Rec",
                "BsmtFin SF 1": 468.0,
                "BsmtFin Type 2": "LwQ",
                "BsmtFin SF 2": 144.0,
                "Bsmt Unf SF": 270.0,
                "Total Bsmt SF": 882.0,
                "Heating": "GasA",
                "Heating QC": "TA",
                "Central Air": "Y",
                "Electrical": "SBrkr",
                "1st Flr SF": 896,
                "2nd Flr SF": 0,
                "Low Qual Fin SF": 0,
                "Gr Liv Area": 896,
                "Bsmt Full Bath": 0.0,
                "Bsmt Half Bath": 0.0,
                "Full Bath": 1,
                "Half Bath": 0,
                "Bedroom AbvGr": 2,
                "Kitchen AbvGr": 1,
                "Kitchen Qual": "TA",
                "TotRms AbvGr": 5,
                "Functional": "Typ",
                "Fireplaces": 0,
                "Fireplace Qu": 'NA',
                "Garage Type": "Attchd",
                "Garage Yr Blt": 1961.0,
                "Garage Finish": "Unf",
                "Garage Cars": 1.0,
                "Garage Area": 730.0,
                "Garage Qual": "TA",
                "Garage Cond": "TA",
                "Paved Drive": "Y",
                "Wood Deck SF": 140,
                "Open Porch SF": 0,
                "Enclosed Porch": 0,
                "3Ssn Porch": 0,
                "Screen Porch": 120,
                "Pool Area": 0,
                "Pool QC": 'NA',
                "Fence": "MnPrv",
                "Misc Feature": 'NA',
                "Misc Val": 0,
                "Mo Sold": 6,
                "Yr Sold": 2010,
                "Sale Type": "WD",
                "Sale Condition": "Normal"
            }
        }


@app.get("/")
def root():
    return {"message": "Welcome to the Ames Housing price prediction API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        # Step 1: Convert input to DataFrame (include raw/extra fields)
        df = pd.DataFrame([payload.dict(by_alias=True)])

        # Step 3: Engineer features like total_sf, bathrooms, etc.
        df = engineer_features(df, CONFIG)

        # Step 2: Clean raw data (renaming, fillna, drop)
        df = clean_raw_data(df, CONFIG)

        # Step 4: Transform with fitted preprocessing pipeline
        df_proc = transform_with_pipeline(df, CONFIG, PIPELINE)

        # Fill missing columns
        missing_cols = set(SELECTED_FEATURES) - set(df_proc.columns)
        for col in missing_cols:
            df_proc[col] = 0

        # Reorder
        df_proc = df_proc[SELECTED_FEATURES]

        # Check final shape
        if df_proc.shape[0] == 0 or df_proc.shape[1] == 0:
            raise HTTPException(
                status_code=500, detail="Preprocessed input is empty — check column names or data content.")

        print("✅ Final input shape:", df_proc.shape)
        print("✅ First row:", df_proc.iloc[0].to_dict())  # confirm content

        # Step 6: Predict
        if df_proc.shape[0] == 0:
            raise HTTPException(
                status_code=500, detail="Input produced zero rows after preprocessing")

        df_final = df_proc[SELECTED_FEATURES]

        prediction = MODEL.predict(df_final)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")
