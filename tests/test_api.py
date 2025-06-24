from app.main import app
import os
import sys
from unittest.mock import patch
from fastapi.testclient import TestClient

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Patch out anything heavy (optional if you don't load artifacts from WandB)
# with patch("scripts.download_from_wandb.download_artifacts", create=True):

client = TestClient(app)

# === Test /health route ===


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

# === Test /predict single ===


def test_predict_single():
    payload = {
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
        "Sale Condition": "Normal",
        "SalePrice": 105000
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert "prediction" in result
    assert isinstance(result["prediction"], float)
