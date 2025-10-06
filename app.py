import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import fetch_california_housing, load_wine

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Data Science Dashboard", layout="wide")
st.title("🔎 Data Science Dashboard – EDA & Quick ML")
st.caption("Learn-by-doing: upload your CSV or use a sample dataset. Run EDA, train simple models, and export predictions.")

# -----------------------------
# Sidebar – Dataset selection
# -----------------------------
st.sidebar.header("1) Choose Dataset")
source = st.sidebar.radio(
    "Data Source",
    ["Upload CSV", "Sample: California Housing (Regression)", "Sample: Wine (Classification)"]
)

@st.cache_data
def load_sample_regression():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"MedHouseVal": "target"}, inplace=True)
    return df, "target"

@st.cache_data
def load_sample_classification():
    data = load_wine(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"target": "target"}, inplace=True)
    return df, "target"

@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    target_col = st.sidebar.text_input("Target column name (required for ML)")
    if uploaded is not None:
        df = read_csv(uploaded)
    else:
        df = None
elif source == "Sample: California Housing (Regression)":
    df, target_col = load_sample_regression()
else:
    df, target_col = load_sample_classification()

# -----------------------------
# Learning note
# -----------------------------
st.info("**Learning note:** Start with EDA to understand shapes, missing values, and distributions before training any model.")

# -----------------------------
# EDA Section
# -----------------------------
st.header("📊 EDA – Exploratory Data Analysis")
if df is None:
    st.warning("Please upload a CSV to continue.")
    st.stop()

st.write("**Shape:**", df.shape)
st.write("**Preview:**")
st.dataframe(df.head())

with st.expander("Columns & dtypes"):
    dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
    st.dataframe(dtypes_df)

with st.expander("Missing values"):
    nulls = df.isnull().sum().sort_values(ascending=False)
    st.dataframe(nulls.to_frame("null_count"))

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Histogram – choose a numeric column")
    if numeric_cols:
        col_choice = st.selectbox("Column", numeric_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col_choice].dropna(), bins=30)
        ax.set_title(f"Histogram: {col_choice}")
        st.pyplot(fig)
    else:
        st.write("No numeric columns found.")

with col2:
    st.subheader("Correlation (numeric only)")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        st.dataframe(corr)
    else:
        st.write("Need at least 2 numeric columns.")

# -----------------------------
# ML Section
# -----------------------------
st.header("🤖 Quick ML Experiments")

if source == "Upload CSV" and not target_col:
    st.warning("Enter the target column name to enable ML.")
else:
    if target_col not in df.columns:
        st.warning(f"Target column '{target_col}' not found in data. ML disabled.")
    else:
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Infer task type
        task_type = st.radio("Task type", ["Auto-detect", "Regression", "Classification"], horizontal=True)
        if task_type == "Auto-detect":
            task = "Regression" if pd.api.types.is_numeric_dtype(y) else "Classification"
        else:
            task = task_type
        st.write(f"**Detected task:** {task}")

        # Train/Test split
        test_size = st.slider("Test size (%)", 10, 40, 20, step=5) / 100.0
        random_state = st.number_input("Random state", value=42, step=1)

        # Preprocessing
        scale_numeric = st.checkbox("Standardize numeric features", value=(task == "Classification"))
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        transformers = []
        if scale_numeric and numeric_features:
            transformers.append(("num", StandardScaler(), numeric_features))
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

        # Model choice
        if task == "Regression":
            model_name = st.selectbox("Model", ["Linear Regression", "Random Forest Regressor"])
            if model_name == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=300, random_state=random_state)
        else:
            model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest Classifier"])
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=200)
            else:
                model = RandomForestClassifier(n_estimators=300, random_state=random_state)

        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

        # Train button
        if st.button("Train model", type="primary"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=(y if task=="Classification" else None)
            )
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            if task == "Regression":
                mae = mean_absolute_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)
                st.subheader("📏 Metrics (Regression)")
                st.write({"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)})
            else:
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")
                st.subheader("📏 Metrics (Classification)")
                st.write({"Accuracy": round(acc, 4), "F1 (macro)": round(f1, 4)})
                cm = confusion_matrix(y_test, y_pred)
                st.write("Confusion Matrix:")
                st.dataframe(pd.DataFrame(cm))

            # Feature importance for RandomForest
            if "Random Forest" in model_name:
                try:
                    importances = pipe.named_steps["model"].feature_importances_
                    # After ColumnTransformer, columns are transformed; approximate names via original sets
                    feature_names = []
                    if scale_numeric and numeric_features:
                        feature_names.extend(numeric_features)
                    feature_names.extend(categorical_features)
                    if len(importances) != len(feature_names):
                        feature_names = [f"feat_{i}" for i in range(len(importances))]
                    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
                    st.subheader("🌳 Feature Importance (Random Forest)")
                    st.dataframe(imp_df)
                except Exception as e:
                    st.caption(f"(Feature importance not available: {e})")

            # Export predictions
            out = pd.DataFrame({"y_true": y_test})
            out["y_pred"] = y_pred
            csv = out.to_csv(index=False).encode()
            st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

            st.success("Training complete. Review metrics and iterate.")

# -----------------------------
# Learning notes footer
# -----------------------------
st.divider()
st.markdown("""
**Learning checkpoints**
- Always inspect missing values and distributions before ML.
- Use a **hold-out** (test set) to estimate generalization. Overfitting = low error on train, high on test.
- Standardization helps for linear/logistic models; tree-based models are scale-invariant.
- Choose metrics by task: MAE/RMSE/R² (regression), Accuracy/F1 (classification).
- Start simple → iterate with better features/models only if needed.
""")
