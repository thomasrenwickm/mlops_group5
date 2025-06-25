# 🏡 Predictive Pricing Engine — MLOps Group 5

![version](https://img.shields.io/badge/version-v2.0-blue)

A production-grade MLOps pipeline for real estate sale price prediction, built for RE/MAX. This project demonstrates a fully automated machine learning system — from data ingestion to deployment — using modern MLOps best practices.

🔗 [GitHub Repository](https://github.com/thomasrenwickm/mlops_group5)  
📊 [W&B Dashboard](https://wandb.ai/mlops-group5/ames_housing_mlops_project?nw=nwuserthomasrenwickm)

---

## 📈 Project Overview

This repository contains a modular MLOps pipeline that predicts house prices using historical data, structured features, and real-time inference through a containerized FastAPI service. It automates the ML lifecycle — including training, tracking, evaluation, and serving — using a modern CI/CD stack.

Built for **RE/MAX**, the goal is to streamline property pricing across franchises and improve trust, consistency, and efficiency in listings.

---

## 🧠 Business Case: RE/MAX

### ❓ Problem

RE/MAX agents often rely on local experience to price homes, which introduces:

- Bias and inconsistency  
- Longer listing times  
- Client distrust  
- Lost revenue opportunities  

> **KPI:** Percentage error between listed price and actual sold price

---

### 🎯 Objectives

- Improve pricing **accuracy and consistency**  
- Reduce **time-to-list**  
- Enhance **agent credibility** with explainable AI  
- Enable **scalable**, automated property valuations  

---

## ✅ Solution Architecture

- 📊 **ML model** trained on housing transaction data  
- ⚙️ **Modular pipeline** with:
  - Data validation
  - Feature engineering
  - Model training and evaluation  
- 🔗 **CI/CD automation** with:
  - GitHub Actions for testing and linting
  - Docker for reproducible builds
  - Render.com for hosted inference
- 📦 **Tracking and observability** via:
  - MLflow for experiment management
  - Hydra for config control
  - Weights & Biases for metrics, logs, and artifacts

---

## 🚀 CI/CD & Deployment (v2.0)

### ✅ Highlights

- **ML Lifecycle Automation**  
  - MLflow for reproducibility  
  - Hydra for modular configs  
  - W&B for experiment tracking and versioning  

- **CI/CD with GitHub Actions**  
  - Automatic test and linting checks on every push or pull request  
  - Guarantees fast feedback and integration reliability  

- **Containerized Deployment**  
  - FastAPI + Docker for scalable, lightweight model serving  
  - Deployed to **Render.com** for public inference  

---

## 🌐 Live Inference API


🔗 `https://mlops-group5-api.onrender.com/docs`

If running locally, use:

```bash
uvicorn app.main:app --reload
```

Then open Swagger UI at:  
👉 [http://127.0.0.1:8000/docs#/default/health_health_get](http://127.0.0.1:8000/docs#/default/health_health_get)

---

### 🧾 Example JSON Input for Prediction

```json
{
  "PID": 526350040,
  "MS SubClass": 20,
  "MS Zoning": "RH",
  "Lot Frontage": 80.0,
  "Lot Area": 11622,
  "Street": "Pave",
  "Alley": null,
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
  "Fireplace Qu": null,
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
  "Pool QC": null,
  "Fence": "MnPrv",
  "Misc Feature": null,
  "Misc Val": 0,
  "Mo Sold": 6,
  "Yr Sold": 2010,
  "Sale Type": "WD",
  "Sale Condition": "Normal"
}
```

---

## 🗂️ Repository Structure

```
├── app/                      # FastAPI app for serving predictions
├── artifacts/                # Inference artifacts (schemas, predictions, etc.)
├── config.yaml               # Main pipeline configuration
├── conda.yml                 # Alternative environment specification
├── data/                     # Datasets managed via DVC
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── inference/
├── data.dvc                  # DVC tracking metadata
├── Dockerfile                # Container setup for deployment
├── environment.yml           # Conda environment specification
├── logs/                     # Validation reports and runtime logs
├── main.py                   # (Legacy) pipeline entry point
├── MLproject                 # MLflow project configuration
├── mlruns/                   # MLflow experiment logs and metadata
├── models/                   # Trained model and pipeline artifacts
├── notebooks/                # Jupyter notebooks for development
├── outputs/                  # Pipeline outputs (grouped by date)
├── pytest.ini                # Pytest test runner configuration
├── README.md                 # Project documentation
├── requirements.txt          # Python dependency list
├── src/                      # Modular pipeline source code
│   ├── data_load/
│   ├── data_validation/
│   ├── features/
│   ├── model/
│   ├── inference/
│   ├── evaluation/
│   ├── preprocess/
│   └── main_legacy.py
├── tests/                    # Unit and integration tests
├── wandb/                    # Weights & Biases experiment tracking logs
└── .github/                  # CI/CD GitHub Actions configuration
```

---

## 🧪 Testing & Code Quality

### 🧪 Run Tests Locally

```bash
pytest tests/
```

You can also test specific files:

```bash
pytest tests/test_data_loader.py
```

Just change the filename to test individual components.

### ✅ CI with GitHub Actions

All tests and linting checks are automatically run on every **push and pull request** using **GitHub Actions**, ensuring continuous integration and code quality.

### 🧹 Linting & Formatting

```bash
black .
flake8 .
```

---

## 💻 Tech Stack

- **Language**: Python 3.10  
- **ML Libraries**: pandas, numpy, scikit-learn, joblib  
- **Experiment Tracking**: MLflow, Weights & Biases  
- **Config Management**: Hydra  
- **API & Serving**: FastAPI, Uvicorn  
- **Containerization**: Docker  
- **CI/CD**: GitHub Actions + Render.com  
- **Testing**: pytest, flake8, black  

---

## 📦 Changelog

### 🚀 v2.0 – CI/CD Pipeline Automation (Pre-release)

- Full CI/CD with GitHub Actions for testing, linting, validation  
- FastAPI + Docker containerization for robust, scalable inference  
- Hosted API on **Render.com**  
- Reproducible ML with MLflow + Hydra + W&B  
- Versioned model artifacts and evaluation logs

🔗 [v2.0 Release](https://github.com/thomasrenwickm/mlops_group5/releases/tag/v2.0-ci%2Fcd-pipeline-automation)

---

### v1.0 – End-to-End ML Pipeline

- Modularized ML pipeline with config and experiment tracking  
- Integrated MLflow, Hydra, and W&B  

🔗 [v1.0 Release](https://github.com/thomasrenwickm/mlops_group5/releases/tag/v1.0)

---

### v0.1.0 – Notebook to Pipeline

- Converted exploratory notebook into modular, testable ML pipeline  

🔗 [v0.1.0 Release](https://github.com/thomasrenwickm/mlops_group5/releases/tag/v0.1.0-notebook-to-mlops-project)

---

## 👥 Team

Developed by:

- Tara Teylouni
- Shadi Alfaraj  
- Yotaro Enomoto  
- Thomas Renwick  
- Massimo Tassinari  
- Joy Zhong

---

## 📬 Feedback & Contributions

- Found a bug? [Open an issue](https://github.com/thomasrenwickm/mlops_group5/issues)  
- Contributions welcome — fork the repo and submit a PR 🚀
