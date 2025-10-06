# Data Science Dashboard (Streamlit)


Interactive dashboard for EDA and quick ML experiments (regression/classification).


## Features
- CSV upload or built-in sample datasets (California Housing, Wine)
- EDA: summary, nulls, dtypes, histograms, correlations
- ML: Linear/Logistic Regression, Random Forest
- Metrics: MAE, RMSE, R² / Accuracy, F1, Confusion Matrix
- Export predictions as CSV

## Screenshots

### 1) Overview & dataset selection
![Overview – dataset selection](images/001-overview.png)

### 2) EDA – Histograms & Correlations
![Histogram – numeric column](images/002-eda-hist.png)
### 3) ML Configuration & Metrics
![ML panel before training](images/003-ml-config.png)
![Regression metrics (MAE, RMSE, R²)](images/004-ml-reg-metrics.png)
![Classification metrics & confusion matrix](images/005-ml-cls-metrics.png)

### 4) Feature importance & export
![Random Forest feature importance](images/006-feature-importance.png)
![Download predictions CSV](images/007-download.png)

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

