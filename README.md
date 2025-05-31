# MLOps Pipeline for Housing Sale Price Predictions

## 📌 Project Overview
This project implements a complete MLOps pipeline, transforming a Jupyter notebook into a modular, testable, and reproducible machine learning system. It follows best practices in software engineering, DevOps, and data science to ensure robust and maintainable solutions.

## 📁 Project Structure
```
├── data/ # Raw, processed, and split data
├── notebooks/ # Exploratory notebooks
├── src/ # Source code for data, features, models
├── tests/ # Unit tests for all modules
├── logs/ # Log files
├── models/ # Serialized models and preprocessing pipelines
├── environment.yml # Conda environment definition
├── config.yaml # Project configuration file
├── .env # Environment variables for secrets
├── .gitignore # Git exclusions
└── README.md # Project documentation
```

## ⚙️ Setup Instructions
1. Clone the repo:
```bash
git clone https://github.com/<your_username>/<repo_name>.git
cd <repo_name>
```
2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate mlops_group5_env
```
3. Run the pipeline:
```bash
python -m src.main
```

## 🧪 Testing
Run unit tests with:
```bash
pytest tests/ --cov=src
```

## 🔧 Tools & Libraries Used
- Python 3.10
- pandas, numpy, openpyxl
- matplotlib, seaborn
- scikit-learn, joblib
- pyyaml, python-dotenv
- pytest, pytest-dotenv, pytest-cov
- black, flake8

## 📊 Business Context
### Client & Stakeholders
- **Target Users:** Home sellers, brokers, and buyers seeking fair price estimations.

### Problem Statement
- Real estate transactions often suffer from pricing opacity. This tool provides a predictive baseline to guide fair decision-making.

### Solution Overview
- A machine learning model trained on historical housing data to predict `SalePrice`, ensuring transparent valuation.

### Motivation
- To create a state-of-the-art MLOps pipeline that supports robust ML models, giving all market participants a fair and scalable pricing tool.

### Expected Benefits
- Provides consistent and data-driven valuations, reduces bias and subjectivity.

### Scalability
- Can be applied across all U.S. markets where similar structured housing data is available.

---

For questions or contributions, feel free to open an issue or pull request!


Collaborators: *Shadi Alfaraj, Yotaro Enomoto, Thomas Renwick, Massimo Tassinari, Joy Zhong*