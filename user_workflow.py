"""
user_workflow.py

Safe, robust implementation of run_workflow for the Streamlit app.

Signature:
    run_workflow(df: pandas.DataFrame, target_column: str = "type", dropna: bool = True) -> dict

Returns a dict with:
 - preds_df: DataFrame of predictions (or None)
 - accuracy: float or None
 - classification_report: str or None
 - figs: dict of matplotlib.figure.Figure objects (confusion, feature_importance, permutation_importance, kmeans)
"""
import os
import re
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


def _normalize_col_name(s: str) -> str:
    """Normalize a column name: remove BOM/zero-width, collapse whitespace/punctuation, lowercase."""
    if s is None:
        return s
    s = str(s)
    s = s.replace("\ufeff", "").replace("\u200b", "")
    s = s.strip()
    # spaces -> underscore, non-word -> underscore, collapse underscores
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.lower()


def _ensure_movies_shows_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has 'movies_shows_by_year'. Detect likely variants, normalize, or compute from release_year.
    Operates on the dataframe whose columns should already be normalized (lowercase + underscores).
    """
    # detect many variant names
    variants = [
        "movies_shows_by_year",
        "movies_shows_per_year",
        "movies_by_year",
        "shows_by_year",
        "movies_shows_byyear",
        "num_movies_by_year",
        "count_by_year",
        "count_per_year",
        "movies_showsbyyear",
    ]
    for v in variants:
        if v in df.columns:
            # rename to canonical if necessary
            if v != "movies_shows_by_year":
                df = df.rename(columns={v: "movies_shows_by_year"})
            return df

    # If missing but release_year present, compute counts per release_year
    if "release_year" in df.columns:
        # coerce release_year to numeric
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        counts = df["release_year"].value_counts(dropna=True)
        df["movies_shows_by_year"] = df["release_year"].map(counts).fillna(0).astype(int)
        return df

    # otherwise leave df untouched; caller will detect missing later
    return df


def run_workflow(df: pd.DataFrame, target_column: str = "type", dropna: bool = True) -> dict:
    out = {"preds_df": None, "accuracy": None, "classification_report": None, "figs": {}}
    try:
        # Work on a copy to avoid surprising caller-side mutations
        df = df.copy()

        # Normalize column names and keep a map back to originals (optional)
        orig_cols = list(df.columns)
        norm_map = { _normalize_col_name(c): c for c in orig_cols }
        df.columns = [ _normalize_col_name(c) for c in orig_cols ]

        # Try to ensure canonical movies_shows_by_year exists (or compute it)
        df = _ensure_movies_shows_by_year(df)

        # Default features (these are normalized names)
        feature_columns = ['title_len', 'release_year', 'runtime', 'imdb_score', 'movies_shows_by_year']

        # If textual columns exist, encode them (originally 'theme' and 'actor')
        for text_col in ('theme', 'actor'):
            if text_col in df.columns and f'{text_col}_encoded' not in df.columns:
                le = LabelEncoder()
                df[f'{text_col}_encoded'] = le.fit_transform(df[text_col].astype(str))
                feature_columns.append(f'{text_col}_encoded')

        # Optionally drop NaNs early (caller chooses via Streamlit UI)
        if dropna:
            df = df.dropna()

        # Check that required features and target exist
        missing = [c for c in feature_columns + [target_column] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Prepare data
        X = df[feature_columns].copy()
        y_raw = df[target_column].astype(str)
        target_le = LabelEncoder()
        y = target_le.fit_transform(y_raw)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Train AdaBoost (Decision Tree base)
        base = DecisionTreeClassifier(max_depth=3, random_state=42)
        ada = AdaBoostClassifier(base_estimator=base, n_estimators=50, learning_rate=1.0, random_state=42)
        ada.fit(X_train, y_train)

        # Predictions & evaluation
        y_pred = ada.predict(X_test)
        y_proba = ada.predict_proba(X_test) if hasattr(ada, "predict_proba") else None
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_le.classes_)

        out["accuracy"] = float(acc)
        out["classification_report"] = report

        preds_df = X_test.copy().reset_index(drop=True)
        preds_df['true_label'] = target_le.inverse_transform(y_test)
        preds_df['predicted_label'] = target_le.inverse_transform(y_pred)
        if y_proba is not None:
            preds_df['confidence'] = np.max(y_proba, axis=1)
        out["preds_df"] = preds_df

        figs = {}

        # Confusion matrix figure
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_le.classes_, yticklabels=target_le.classes_, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        figs["confusion"] = fig_cm

        # AdaBoost feature importance
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
        try:
            perm_imp = permutation_importance(ada, X_test, y_test, n_repeats=10, random_state=42)
            sorted_idx = perm_imp.importances_mean.argsort()
            fig_perm, ax = plt.subplots(figsize=(8,4))
            ax.boxplot(perm_imp.importances[sorted_idx].T, vert=False, labels=np.array(feature_columns)[sorted_idx])
            ax.set_title("Permutation Feature Importance")
            plt.tight_layout()
            figs["permutation_importance"] = fig_perm
        except Exception:
            figs["permutation_importance"] = None

        # Robust k-means: select available numeric features from requested list
        requested_kmeans = ['title_len', 'release_year', 'runtime', 'imdb_score', 'movies_shows_by_year']
        available = [c for c in requested_kmeans if c in df.columns]
        numeric_cols = []
        for c in available:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].isna().all():
                continue
            if df[c].isna().any():
                med = df[c].median()
                df[c] = df[c].fillna(med)
            numeric_cols.append(c)

        if len(numeric_cols) >= 2:
            kf = df[numeric_cols]
            try:
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(kf)
                df['kmeans_cluster'] = clusters
                # use first two numeric cols for scatter
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                fig_km, ax = plt.subplots(figsize=(7,5))
                sc = ax.scatter(df[x_col], df[y_col], c=df['kmeans_cluster'], cmap='viridis', alpha=0.7)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title('k-Means Clusters (4 clusters)')
                plt.colorbar(sc, ax=ax)
                plt.tight_layout()
                figs["kmeans"] = fig_km
            except Exception:
                figs["kmeans"] = None
        else:
            figs["kmeans"] = None

        out["figs"] = figs
        return out

    except Exception as e:
        # bubble up meaningful error to the Streamlit UI
        raise


# End of file
