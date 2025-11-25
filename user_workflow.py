"""
user_workflow.py

Exposes run_workflow(df, target_column='type', dropna=True) -> dict

Returned dict keys:
 - preds_df: pandas.DataFrame with predictions (or None)
 - accuracy: float or None
 - classification_report: str or None
 - figs: dict of matplotlib.figure.Figure objects (confusion, feature_importance, permutation_importance, kmeans)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

sns.set_style("whitegrid")

def run_workflow(df: pd.DataFrame, target_column: str = "type", dropna: bool = True) -> dict:
    out = {"preds_df": None, "accuracy": None, "classification_report": None, "figs": {}}
    try:
        # Basic exploration (prints are useful in logs)
        # print("Dataset shape:", df.shape)
        # print(df.head())

        if dropna:
            df = df.dropna()

        # Default feature set — change if your CSV differs
        feature_columns = ['title_len', 'release_year', 'runtime', 'imdb_score', 'movies_shows_by_year']

        # Encode 'theme' and 'actor' if available
        encoders = {}
        for col in ('theme', 'actor'):
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                feature_columns.append(f'{col}_encoded')

        # Verify
        missing = [c for c in feature_columns + [target_column] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X = df[feature_columns]
        y_raw = df[target_column].astype(str)

        target_le = LabelEncoder()
        y = target_le.fit_transform(y_raw)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # AdaBoost
        base = DecisionTreeClassifier(max_depth=3, random_state=42)
        ada = AdaBoostClassifier(base_estimator=base, n_estimators=50, learning_rate=1.0, random_state=42)
        ada.fit(X_train, y_train)

        y_pred = ada.predict(X_test)
        y_proba = ada.predict_proba(X_test) if hasattr(ada, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_le.classes_)

        out["accuracy"] = float(acc)
        out["classification_report"] = report

        # Predictions dataframe
        preds_df = X_test.copy().reset_index(drop=True)
        preds_df['true_label'] = target_le.inverse_transform(y_test)
        preds_df['predicted_label'] = target_le.inverse_transform(y_pred)
        if y_proba is not None:
            preds_df['confidence'] = np.max(y_proba, axis=1)
        out["preds_df"] = preds_df

        # Figures
        figs = {}

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_le.classes_, yticklabels=target_le.classes_, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        figs["confusion"] = fig_cm

        # Feature importance (AdaBoost)
        importances = ada.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig_fi, ax = plt.subplots(figsize=(8,4))
        ax.bar(range(X.shape[1]), importances[indices])
        ax.set_xticks(range(X.shape[1]))
        ax.set_xticklabels([feature_columns[i] for i in indices], rotation=45, ha="right")
        ax.set_title("AdaBoost Feature Importance (Gini)")
        plt.tight_layout()
        figs["feature_importance"] = fig_fi

        # Permutation importance
        perm_imp = permutation_importance(ada, X_test, y_test, n_repeats=10, random_state=42)
        sorted_idx = perm_imp.importances_mean.argsort()
        fig_perm, ax = plt.subplots(figsize=(8,4))
        ax.boxplot(perm_imp.importances[sorted_idx].T, vert=False, labels=np.array(feature_columns)[sorted_idx])
        ax.set_title("Permutation Feature Importance")
        plt.tight_layout()
        figs["permutation_importance"] = fig_perm

     # --- BEGIN: robust handling for movies_shows_by_year and k-means features ---

# 1) Normalize header names to handle BOMs, spaces, weird chars
import re
def _normalize(s: str) -> str:
    if s is None:
        return s
    s = str(s).replace('\ufeff', '').replace('\u200b', '')
    s = s.strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^\w]', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.lower()

orig_cols = list(df.columns)
norm_map = { _normalize(c): c for c in orig_cols }
# rename df columns to normalized names (temporary)
df.columns = [_normalize(c) for c in orig_cols]

# 2) If an alternate name exists for movies_shows_by_year, map it to canonical
candidates = ["movies_shows_by_year","movies_shows_per_year","movies_by_year","shows_by_year",
              "movies_shows_byyear","num_movies_by_year","count_by_year","count_per_year"]
for cand in candidates:
    if cand in df.columns and cand != "movies_shows_by_year":
        df = df.rename(columns={cand: "movies_shows_by_year"})
        break

# 3) If movies_shows_by_year still missing but release_year exists, compute counts-per-year
if "movies_shows_by_year" not in df.columns and "release_year" in df.columns:
    # ensure release_year is numeric
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    counts = df["release_year"].value_counts(dropna=True)
    df["movies_shows_by_year"] = df["release_year"].map(counts).fillna(0).astype(int)
    # optional: log so user sees action
    print("Computed movies_shows_by_year from release_year (counts per year).")

# 4) Restore nicer (original) column names if you want, or proceed with normalized names.
# If the rest of your code expects original names, map back:
# Build reverse mapping for original column strings
reverse_map = {}
for norm, orig in norm_map.items():
    reverse_map[norm] = orig
# If you prefer to keep normalized names, skip this step.
# If you want to revert others back to original for compatibility:
# df = df.rename(columns={norm: reverse_map.get(norm, norm) for norm in df.columns})

# --- prepare numeric k-means features robustly ---
requested_kmeans = ['title_len', 'release_year', 'runtime', 'imdb_score', 'movies_shows_by_year']
# choose available ones (normalized)
available_kmeans = [c for c in requested_kmeans if c in df.columns]

# coerce to numeric and handle missing values
clean_kmeans = []
for c in available_kmeans:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    # fill small number of missing values with median to avoid dropping too many rows
    if df[c].isna().any():
        med = df[c].median()
        if pd.isna(med):
            # if median is NaN (column all NaN), skip the column
            continue
        df[c] = df[c].fillna(med)
    clean_kmeans.append(c)

# If we have at least 2 numeric features, run k-means; otherwise skip gracefully
if len(clean_kmeans) >= 2:
    kf = df[clean_kmeans]
    from sklearn.cluster import KMeans
    try:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(kf)
        df['kmeans_cluster'] = clusters
        # plotting below will use whichever columns exist (choose first two for scatter)
        x_col, y_col = clean_kmeans[0], clean_kmeans[1]
        # create the scatter figure as before, but use x_col, y_col
        fig_km, ax = plt.subplots(figsize=(10,7))
        scatter = ax.scatter(df[x_col], df[y_col], c=df['kmeans_cluster'], cmap='viridis', alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title('k-Means Clusters (4 clusters)')
        fig_km.colorbar = plt.colorbar(scatter, ax=ax)
        # save or attach fig_km for Streamlit display
        # e.g., figs["kmeans"] = fig_km
    except Exception as e:
        print("k-means skipped due to error:", e)
        # set figs["kmeans"] = None or continue
else:
    print(f"Skipping k-means: need >=2 numeric features, found {len(clean_kmeans)}: {clean_kmeans}")

# --- END robust k-means handling ---

        # k-Means scatter (if numeric columns present)
        kmeans_features = ['title_len', 'release_year', 'runtime', 'imdb_score', 'movies_shows_by_year']
        if all(col in df.columns for col in kmeans_features):
            kf = df[kmeans_features]
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(kf)
            df['kmeans_cluster'] = clusters
            fig_km, ax = plt.subplots(figsize=(7,5))
            sc = ax.scatter(df['imdb_score'], df['runtime'], c=df['kmeans_cluster'], cmap='viridis', alpha=0.7)
            ax.set_xlabel('IMDb Score')
            ax.set_ylabel('Runtime (min)')
            ax.set_title('k-Means Clusters (4 clusters)')
            plt.colorbar(sc, ax=ax)
            plt.tight_layout()
            figs["kmeans"] = fig_km
        else:
            figs["kmeans"] = None

        out["figs"] = figs

        return out

    except Exception as e:
        # Reraise for the Streamlit app to display the error
        raise
