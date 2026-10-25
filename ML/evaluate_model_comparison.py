"""
evaluate_model_comparison.py
-------------------------------
Trains and fairly compares THREE candidate classifiers on identical
cross-validation folds, then selects and saves the best one as the
production model. This directly answers "why Random Forest?" with
evidence instead of assumption -- exactly what a data science reviewer
expects to see before trusting a single model's numbers.

MODELS COMPARED:
    1. Logistic Regression  -- simple, fast, highly interpretable linear
       baseline. If a complex model can't beat this, the complexity isn't
       earning its keep.
    2. Random Forest        -- the model already in production. An
       ensemble of decision trees, good at capturing non-linear symptom
       interactions.
    3. Gradient Boosting    -- a stronger ensemble method (HistGradientBoosting,
       scikit-learn's fast modern implementation) that often outperforms
       Random Forest on tabular data by correcting errors iteratively.

METHODOLOGY:
    All three models are evaluated on the EXACT SAME stratified 5-fold
    cross-validation split (same random seed), so the comparison is fair --
    no model gets an easier or harder split than another. For each model
    we report: CV accuracy (mean +/- std), macro precision/recall/F1, and
    macro ROC-AUC (one-vs-rest). The winner is selected by CV mean
    accuracy, with macro F1 as an explicit tie-breaker if two models are
    within 0.5 percentage points of each other (documented, not silent).

OUTPUT:
    ML/evaluation/model_comparison.json  -- full comparison table, all 3 models
    ML/disease_classifier.joblib          -- retrained WINNING model (replaces
                                              whatever was there before)
    ML/symptom_features.joblib            -- unchanged, same vocabulary
    ML/disease_keph_mapping.joblib        -- unchanged

Usage:
    python evaluate_model_comparison.py
"""

import json
import os
import time

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from train_model import load_raw_rows, build_symptom_vocabulary, vectorize, DATA_PATH, RANDOM_SEED

OUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS = 5
TIE_BREAK_MARGIN = 0.005  # 0.5 percentage points


def get_candidate_models():
    """Returns {name: zero-arg factory function producing a FRESH unfitted
    estimator}. Using factories (not pre-built instances) guarantees every
    CV fold trains a clean model with no leftover fitted state from a
    previous fold -- kept as a function so hyperparameters are easy to
    find and adjust in one place."""
    return {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED,
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Gradient Boosting": lambda: HistGradientBoostingClassifier(
            random_state=RANDOM_SEED, max_iter=150,
        ),
    }


def evaluate_model(name, model_factory, X, y, skf, classes):
    """
    model_factory: a zero-arg callable returning a FRESH unfitted model
    instance (so each fold gets a clean model, never reusing fitted state).

    Runs ONE manual cross-validation loop -- each fold fits the model
    exactly once, and fold accuracy, out-of-fold predictions, and
    out-of-fold probabilities are all extracted from that single fit.
    (An earlier version of this script called cross_val_score AND
    cross_val_predict AND cross_val_predict(method='predict_proba')
    separately -- three full training passes per model, needlessly slow.
    This version fits each fold once.)
    """
    print(f"\nEvaluating {name}...")
    t0 = time.time()

    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    fold_accuracies = []
    y_pred_oof = np.empty(len(y), dtype=object)
    y_proba_oof = np.zeros((len(y), n_classes))

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])

        fold_pred = model.predict(X[test_idx])
        fold_acc = accuracy_score(y[test_idx], fold_pred)
        fold_accuracies.append(fold_acc)
        y_pred_oof[test_idx] = fold_pred

        fold_proba = model.predict_proba(X[test_idx])
        # model.classes_ may be a subset/different order than our global
        # `classes` list if a fold happens to miss a rare class -- map
        # columns back to the global class index explicitly rather than
        # assuming column order matches.
        for col, cls in enumerate(model.classes_):
            y_proba_oof[test_idx, class_to_idx[cls]] = fold_proba[:, col]

    cv_scores = np.array(fold_accuracies)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred_oof, average="macro", zero_division=0
    )

    roc_auc = None
    try:
        y_bin = label_binarize(y, classes=classes)
        roc_auc = roc_auc_score(y_bin, y_proba_oof, average="macro", multi_class="ovr")
    except Exception as e:
        print(f"  (Could not compute ROC-AUC for {name}: {e})")

    elapsed = time.time() - t0
    result = {
        "model": name,
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "cv_fold_scores": [round(float(s), 4) for s in cv_scores],
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "roc_auc_macro_ovr": round(float(roc_auc), 4) if roc_auc is not None else None,
        "training_seconds": round(elapsed, 1),
    }
    print(f"  CV accuracy: {result['cv_accuracy_mean']:.4f} +/- {result['cv_accuracy_std']:.4f}  "
          f"(macro F1: {result['macro_f1']:.4f}, took {elapsed:.1f}s)")
    return result


def select_winner(results):
    """Highest CV accuracy wins; macro F1 breaks ties within TIE_BREAK_MARGIN."""
    best = max(results, key=lambda r: r["cv_accuracy_mean"])
    close_contenders = [
        r for r in results
        if best["cv_accuracy_mean"] - r["cv_accuracy_mean"] <= TIE_BREAK_MARGIN
    ]
    if len(close_contenders) > 1:
        winner = max(close_contenders, key=lambda r: r["macro_f1"])
        reason = (
            f"'{winner['model']}' selected: within {TIE_BREAK_MARGIN*100:.1f} points of the top "
            f"CV accuracy alongside {[r['model'] for r in close_contenders if r is not winner]}, "
            f"won on macro F1 tie-break ({winner['macro_f1']} vs others)."
        )
    else:
        winner = best
        reason = f"'{winner['model']}' selected: highest CV accuracy ({winner['cv_accuracy_mean']:.4f})."
    return winner, reason


def main():
    print("=" * 70)
    print("MULTI-MODEL COMPARISON")
    print("=" * 70)

    header, rows = load_raw_rows(DATA_PATH)
    vocab = build_symptom_vocabulary(rows)
    X, y = vectorize(rows, vocab)
    classes = sorted(set(y))
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {len(classes)} classes")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    model_factories = get_candidate_models()

    results = []
    for name, factory in model_factories.items():
        results.append(evaluate_model(name, factory, X, y, skf, classes))

    winner_result, reason = select_winner(results)
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<22}{'CV Accuracy':<18}{'Macro F1':<12}{'ROC-AUC':<10}")
    for r in sorted(results, key=lambda r: -r["cv_accuracy_mean"]):
        marker = " <-- SELECTED" if r["model"] == winner_result["model"] else ""
        acc_str = f"{r['cv_accuracy_mean']:.4f} +/- {r['cv_accuracy_std']:.4f}"
        print(f"{r['model']:<22}{acc_str:<18}{r['macro_f1']:<12}{r['roc_auc_macro_ovr']:<10}{marker}")
    print(f"\n{reason}")

    # Retrain the winning model on ALL data (not just CV folds) for production use
    print(f"\nRetraining '{winner_result['model']}' on full dataset for deployment...")
    winning_model = model_factories[winner_result["model"]]()
    winning_model.fit(X, y)

    joblib.dump(winning_model, os.path.join(os.path.dirname(__file__), "disease_classifier.joblib"))
    print("Saved -> disease_classifier.joblib")

    comparison_output = {
        "models_compared": results,
        "selected_model": winner_result["model"],
        "selection_reason": reason,
        "n_folds": N_FOLDS,
        "dataset": {
            "total_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": len(classes),
        },
    }
    out_path = os.path.join(OUT_DIR, "model_comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison_output, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
