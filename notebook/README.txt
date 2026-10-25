HOW TO RUN THE NOTEBOOK
=======================
Keep all four files in the SAME folder:
  - Healthcare_Model_Development.ipynb   (the notebook, already executed once with real outputs saved)
  - train_model.py                        (project's own vectorization code, imported by the notebook)
  - dataset.csv                           (augmented training dataset, 6,150 rows)
  - dataset_real_kaggle_raw.csv           (original raw Kaggle dataset, used in the data-quality section)

Then in your activated venv:
  pip install notebook scikit-learn pandas numpy matplotlib seaborn
  jupyter notebook Healthcare_Model_Development.ipynb

Runs in about 30-60 seconds by default. The two slow cells (Random Forest and Gradient Boosting
hyperparameter searches) are skipped by default and show their already-captured real results instead.
To re-run those live (a few extra minutes), set RUN_SLOW_CELLS = True in the Setup cell.
