# IMDB-Rating Streamlit App

This repository contains a Streamlit app to run the IMDB workflow (preprocessing, AdaBoost training/evaluation, feature importances, k-means visualization) on a CSV dataset.

Files included
- streamlit_app.py — Streamlit frontend and runner
- user_workflow.py — workflow implementation (exposes run_workflow(df))
- requirements.txt — dependencies (including streamlit)
- .gitignore — common ignores
- LICENSE — MIT

Quick start (local)
1. Place your CSV dataset in the `data/` folder of the repository (create `data/` if missing).
   - The CSV should include the columns used by the workflow:
     index, id, title_len, type, description, release_year, runtime, imdb_score, theme, actor, movies_shows_by_year
   - If your column names differ, you can change them via the app UI.

2. Create and activate a virtual environment, install deps:
   python3 -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. Run the Streamlit app:
   streamlit run streamlit_app.py

How to use the app
- In the sidebar choose a dataset:
  - "Repo data folder" will list CSV files under `./data/` (use this if you uploaded your dataset to the repo).
  - "Upload file" allows you to upload a CSV at runtime.
- Set options (target column, whether to drop NaNs).
- Click "Run workflow". The app will:
  - Show dataset preview and basic info
  - Train AdaBoost (DecisionTree base) to predict the chosen target
  - Show accuracy and classification report
  - Display confusion matrix, feature importance, permutation importance, and k-means scatter (if numeric columns available)
  - Provide a download button for predictions CSV

Notes
- Avoid committing very large datasets to GitHub. For small datasets it's fine to keep them in `data/`.
- If your CSV has different column names or you want to use different features, adjust the feature selection in `user_workflow.py` or change the target field from the sidebar.
- The app saves temporary outputs in the `outputs/` folder (created during run).

If you'd like, I can:
- Adjust the UI to expose more hyperparameters (n_estimators, max_depth, n_clusters).
- Add a sample generator to create a small demo CSV in `data/` for quick testing.
