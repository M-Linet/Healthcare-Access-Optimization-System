# Extended model search

A small manual search run after the main model comparison, to check whether the default settings
were leaving accuracy on the table. It uses the same data, split and 5-fold cross-validation as
`ML/train_model.py`.

- `search.py` tries Logistic Regression (C from 0.1 to 100), Naive Bayes, SVM, k-NN, a small neural
  network, Random Forest, Extra Trees, gradient boosting and XGBoost. Results are in `search_results.json`.
- `ens.py` and `final_eval.py` try simple ensembles and check the best candidates on the held-out test split.

Result: the best candidates landed within about half a point of Logistic Regression, which is inside
the fold-to-fold spread (about 0.5 points), so Logistic Regression with default settings was kept.
These scores come from the augmented dataset and are probably optimistic.

Run from the `ML` folder: `python extended_search/search.py`. XGBoost is needed for `search.py`
(`pip install xgboost`); it is not required to run the app.
