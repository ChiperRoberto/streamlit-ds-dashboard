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

### 2) EDA – histograme și corelații
![Histogram – numeric column](images/002-eda-hist.png)
![Correlation matrix](images/003-eda-corr.png)

### 3) Configurare ML & metrici
![ML panel before training](images/004-ml-config.png)
![Regression metrics (MAE, RMSE, R²)](images/005-ml-reg-metrics.png)
![Classification metrics & confusion matrix](images/006-ml-cls-metrics.png)

### 4) Feature importance & export
![Random Forest feature importance](images/007-feature-importance.png)
![Download predictions CSV](images/008-download.png)

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

