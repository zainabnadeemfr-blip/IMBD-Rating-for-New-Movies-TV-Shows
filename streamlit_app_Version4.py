#!/usr/bin/env python3
"""
Streamlit app to run the IMDB workflow.

Usage:
  streamlit run streamlit_app.py
"""
import os
import io
import pandas as pd
import streamlit as st

from user_workflow import run_workflow

st.set_page_config(page_title="IMDB Workflow", layout="wide")

st.title("IMDB Movies & TV Shows — Workflow (AdaBoost + k-Means)")

# Sidebar: dataset selection
st.sidebar.header("Dataset")
source = st.sidebar.radio("Choose dataset source", ("Repo data folder", "Upload file"))

csv_path = None
uploaded_df = None

if source == "Repo data folder":
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        st.sidebar.warning("No CSV files found in ./data/. Upload a CSV or add one to the data folder in your repo.")
    selected = st.sidebar.selectbox("Select CSV from data/", [""] + csv_files)
    if selected:
        csv_path = os.path.join(data_dir, selected)
elif source == "Upload file":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded CSV: {e}")

# Workflow options
st.sidebar.header("Options")
target_column = st.sidebar.text_input("Target column (predict)", value="type")
dropna = st.sidebar.checkbox("Drop rows with any NaN (default)", value=True)
run_button = st.sidebar.button("Run workflow")

# Show dataset preview
if csv_path:
    try:
        df = pd.read_csv(csv_path)
        st.subheader("Dataset preview")
        st.write(f"Loaded from: {csv_path}")
        st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Error reading CSV at {csv_path}: {e}")
        df = None
elif uploaded_df is not None:
    df = uploaded_df
    st.subheader("Uploaded dataset preview")
    st.dataframe(df.head(10))
else:
    df = None

# Run workflow when requested
if run_button:
    if df is None:
        st.error("No dataset selected. Choose a CSV from ./data/ or upload a file.")
    else:
        st.info("Running workflow — this may take a moment.")
        try:
            # Pass options via df copy if needed; run_workflow returns results (see user_workflow.py)
            results = run_workflow(df.copy(), target_column=target_column, dropna=dropna)
            # results expected to include keys: preds_df, accuracy, classification_report, figs(dict)
            st.success("Workflow completed.")

            # Accuracy & classification report
            st.subheader("Evaluation")
            st.write(f"Accuracy: **{results.get('accuracy', 'N/A')}**")
            st.text(results.get("classification_report", "No classification report available."))

            # Predictions table and download
            if results.get("preds_df") is not None:
                st.subheader("Predictions (sample)")
                st.dataframe(results["preds_df"].head(10))
                csv_bytes = results["preds_df"].to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="adaboost_predictions.csv", mime="text/csv")

            # Plots
            figs = results.get("figs", {})
            if "confusion" in figs and figs["confusion"] is not None:
                st.subheader("Confusion Matrix")
                st.pyplot(figs["confusion"])
            if "feature_importance" in figs and figs["feature_importance"] is not None:
                st.subheader("Feature Importance (Gini)")
                st.pyplot(figs["feature_importance"])
            if "permutation_importance" in figs and figs["permutation_importance"] is not None:
                st.subheader("Permutation Importance")
                st.pyplot(figs["permutation_importance"])
            if "kmeans" in figs and figs["kmeans"] is not None:
                st.subheader("k-Means scatter")
                st.pyplot(figs["kmeans"])

        except Exception as e:
            st.error(f"Workflow failed: {e}")