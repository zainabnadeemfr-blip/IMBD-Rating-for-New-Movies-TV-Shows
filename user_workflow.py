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
