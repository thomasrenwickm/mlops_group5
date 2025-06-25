# Predictive Pricing Engine — MLOps Group 5

## Project Overview
This repository contains a production grade MLOps pipeline that automates housing sale price predictions. Built using modular code, testing frameworks, and pipeline orchestration best practices, the project transforms a data science notebook into a deployable ML solution.

The solution was developed for RE/MAX, one of the world’s leading real estate franchises, to address inconsistencies in property pricing and support faster, more data driven listing strategies.

---

## Business Case: RE/MAX
### Problem
RE/MAX agents often rely on experience or local knowledge when pricing properties. While useful, this introduces bias, inconsistency across franchises, and slows down the listing process. As a result:
- Sellers may lose trust due to mispricing.
- Buyers face uncertainty.
- Time-to-market increases.
- Revenue potential is not fully captured.

**KPI Tracked:**  
Percentage error between listed price and actual sold price.

---

### Objectives
- Improve pricing accuracy and consistency across RE/MAX listings.
- Reduce time-to-price, accelerating the listing process.
- Strengthen agent credibility and consumer trust.
- Enable scalable, explainable pricing tools for broad adoption.

---

### Solution
- A trained ML model that predicts sale prices based on historical transactions, location, property characteristics, and textual descriptions.
- A modular MLOps pipeline including preprocessing, feature selection, model training, and evaluation.
- Output includes explainable predictions and metrics, intended to be integrated into RE/MAX’s CRM and listing platforms.

---

### Scalability & Extensions
- Rollout can begin with pilot regions and expand franchise by franchise.
- The model can be adapted to different geographies or enhanced with third-party data (walkability, schools, zoning).
- Future opportunities:
  - Buyer-home matchmaking based on preference similarity.
  - Market forecasting dashboards for investment strategies.
  - Instant offers using predictive valuations (iBuyer model).

---

## Repository Structure
```
├── data/ # Input datasets (e.g., new_data.csv)
├── models/ # Trained models and pipelines (PKL, JSON)
├── notebooks/ # Development notebook
├── src/ # Main pipeline source code
│ ├── data_load/ # Data ingestion logic
│ ├── data_validation/ # Schema and quality checks
│ ├── features/ # Feature extraction and selection
│ ├── model/ # Model training and serialization
│ ├── inference/ # Prediction module
│ ├── evaluation/ # Model evaluation and metrics
│ └── preprocess/ # Data preprocessing pipeline
├── tests/ # Unit tests per module
├── config.yaml # Configuration settings
├── environment.yml # Conda environment definition
├── pytest.ini # Pytest configuration
└── README.md # Project documentation
```
---

## Setup & Execution
### 1. Clone the repository
```bash
git clone https://github.com/<your_username>/mlops_group5.git
cd mlops_group5
```
### 2. Set up environment
```bash
conda env create -f environment.yml
conda activate mlops_group5_env
```
### 3. Run the pipeline:
```bash
python -m src.main
```
---

## Testing
Run unit tests with:
```bash
pytest tests/ --cov=src
```

---

## Tech Stack
- Python 3.10
- pandas, numpy, scikit-learn
- joblib, openpyxl, pyyaml
- pytest, coverage, flake8, black
- Conda for dependency management

---

## Team 
This project was completed by:
- Shadi Alfaraj
- Yotaro Enomoto
- Thomas Renwick
- Massimo Tassinari
- Joy Zhong

--- 

For contributions, please open a pull request.
For feedback or questions, contact any team member or open an issue.

## Changelog

### v0.1.0-notebook-to-mlops-project
- Functional pipeline before integrating MLflow Projects, Hydra, and W&B
- For reference or rollback, check [this release/tag](https://github.com/thomasrenwickm/mlops_group5/releases/tag/v0.1.0-notebook-to-mlops-project)

### v1.0
- Fully automated pipeline using **MLflow Projects**, **Hydra**, and **Weights & Biases**
- Supports end-to-end training, evaluation, and inference with traceable experiments and artifacts
- For reference or rollback, check [this release/tag](https://github.com/thomasrenwickm/mlops_group5/releases/tag/v1.0)

### v2.0-ci/cd-pipeline-automation

- Introduced CI/CD pipeline for model deployment and delivery
- **CI** powered by GitHub Actions
- **CD** implemented with:
  - Docker for containerization
  - FastAPI for serving the model via REST API
  - Render.com for cloud hosting the API
- Improves automation, reproducibility, and accessibility for production-ready ML workflows
- For reference or rollback, check [this release/tag](https://github.com/thomasrenwickm/mlops_group5/releases/tag/v2.0-ci%2Fcd-pipeline-automation)
