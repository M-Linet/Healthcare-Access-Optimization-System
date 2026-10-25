"""
evaluate_model.py (fixed)
----------------------------
FIX: this script previously ALWAYS hardcoded and retrained a fresh
RandomForestClassifier, completely independent of whatever
evaluate_model_comparison.py actually selected and saved as the
production disease_classifier.joblib. That meant metrics.json could
describe a different algorithm than the one actually deployed and
answering live predictions -- a real inconsistency for an academic
defense, where the "Model Performance" page must describe the model
that is actually running.

This version reads ML/evaluation/model_comparison.json (if it exists)
to find out which model evaluate_model_comparison.py selected, and
evaluates THAT model with the SAME hyperparameters, so metrics.json
always describes the model that's actually deployed. If no comparison
has been run yet, it falls back to the original Random Forest default
(with a clear note in the output) so this script still works standalone.

FIX 2: feature importance was hardcoded to clf.feature_importances_,
which only exists on tree-based models (Random Forest, Gradient
Boosting). Linear models like Logistic Regression have .coef_ instead --
a fundamentally different kind of "importance" (coefficient magnitude,
not impurity reduction). This version handles both, and labels the
result with which method was actually used so the frontend can explain
it honestly rather than implying they're the same thing.

Usage:
    python evaluate_model.py
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from train_model import (
    load_raw_rows, build_symptom_vocabulary, vectorize,
    manual_train_test_split, DATA_PATH, RANDOM_SEED,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

COMPARISON_PATH = os.path.join(OUT_DIR, "model_comparison.json")

# Must exactly match evaluate_model_comparison.py's get_candidate_models(),
# so re-instantiating a model by name here produces identical hyperparameters
# to what was actually compared and selected.
MODEL_BUILDERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_iter=150),
}
DEFAULT_MODEL_NAME = "Random Forest"  # used only if no comparison has ever been run


def get_active_model_name():
    """
    Returns the model name to evaluate. Reads model_comparison.json's
    `selected_model` if it exists; otherwise falls back to the original
    Random Forest default and prints a note.
    """
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH) as f:
            comparison = json.load(f)
        selected = comparison.get("selected_model")
        if selected in MODEL_BUILDERS:
            print(f"Using model selected by evaluate_model_comparison.py: '{selected}'")
            return selected
        else:
            print(f"WARNING: model_comparison.json names '{selected}', which isn't in "
                  f"MODEL_BUILDERS. Falling back to default ({DEFAULT_MODEL_NAME}).")
    else:
        print(f"No model_comparison.json found -- run evaluate_model_comparison.py first "
              f"for this to reflect an actual model-selection decision. "
              f"Falling back to default ({DEFAULT_MODEL_NAME}).")
    return DEFAULT_MODEL_NAME


def extract_feature_importance(clf, vocab, top_n=20):
    """
    Returns (method_name, [{"symptom":..., "importance":...}, ...]).
    Handles both tree-based (feature_importances_) and linear
    (coef_) models, clearly labeling which kind of "importance" this is --
    they are NOT directly comparable across model types.
    """
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        method = "gini_importance"  # impurity-based, tree models
    elif hasattr(clf, "coef_"):
        # coef_ shape is (n_classes, n_features) for multiclass linear models.
        # Mean absolute coefficient magnitude across classes as a single
        # per-feature score -- a common, defensible summary, but a genuinely
        # different concept from impurity-based importance (direction-free,
        # sensitive to feature scaling, and specific to a linear decision
        # boundary rather than a tree's splits).
        importances = np.mean(np.abs(clf.coef_), axis=0)
        method = "mean_abs_coefficient"
    else:
        return "unavailable", []

    order = np.argsort(importances)[::-1][:top_n]
    top_features = [(vocab[i], float(importances[i])) for i in order]
    return method, [{"symptom": name, "importance": round(val, 5)} for name, val in top_features]


def main():
    print("=" * 70)
    print("MODEL EVALUATION — Healthcare Access Optimization System")
    print("=" * 70)

    model_name = get_active_model_name()
    model_builder = MODEL_BUILDERS[model_name]

    # --- Load & prepare data (identical pipeline to train_model.py) ---
    header, rows = load_raw_rows(DATA_PATH)
    vocab = build_symptom_vocabulary(rows)
    X, y = vectorize(rows, vocab)
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y)

    classes = sorted(set(y))
    n_classes = len(classes)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} symptom features, {n_classes} disease classes")
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    metrics = {}

    # -----------------------------------------------------------------
    # 1. Train the active model (same hyperparameters as production)
    # -----------------------------------------------------------------
    clf = model_builder()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    _p, _r, _f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    test_precision_macro, test_recall_macro, test_f1_macro = _p, _r, _f1

    train_accuracy = accuracy_score(y_train, clf.predict(X_train))

    metrics["model_config"] = {
        "algorithm": type(clf).__name__,
        "selected_by": "evaluate_model_comparison.py" if os.path.exists(COMPARISON_PATH) else "default (comparison not yet run)",
        "hyperparameters": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                             for k, v in clf.get_params().items()},
        "random_state": RANDOM_SEED,
    }

    metrics["accuracy"] = {
        "test": round(float(test_accuracy), 4),
        "train": round(float(train_accuracy), 4),
        "overfitting_gap": round(float(train_accuracy - test_accuracy), 4),
    }
    print(f"\n[1/6] Held-out test set: accuracy={test_accuracy:.4f}, macro-F1={test_f1_macro:.4f}")
    print(f"       Train accuracy={train_accuracy:.4f} (gap: {train_accuracy - test_accuracy:.4f})")

    # -----------------------------------------------------------------
    # 2. Stratified 5-fold cross-validation
    # -----------------------------------------------------------------
    print("\n[2/6] Running stratified 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_clf = model_builder()
    cv_scores = cross_val_score(cv_clf, X, y, cv=skf, scoring="accuracy")
    metrics["accuracy"]["cv_5fold_mean"] = round(float(cv_scores.mean()), 4)
    metrics["accuracy"]["cv_5fold_std"] = round(float(cv_scores.std()), 4)
    metrics["accuracy"]["cv_5fold_scores"] = [round(float(s), 4) for s in cv_scores]
    print(f"       CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # -----------------------------------------------------------------
    # 3. Per-class classification report
    # -----------------------------------------------------------------
    print("\n[3/6] Generating per-class classification report...")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, zero_division=0
    )
    metrics["macro_avg"] = {
        "precision": round(float(test_precision_macro), 4),
        "recall": round(float(test_recall_macro), 4),
        "f1": round(float(test_f1_macro), 4),
    }
    metrics["weighted_avg"] = {
        "precision": round(float(np.average(precision, weights=support)), 4),
        "recall": round(float(np.average(recall, weights=support)), 4),
        "f1": round(float(np.average(f1, weights=support)), 4),
    }
    report_text = classification_report(y_test, y_pred, zero_division=0)
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report_text)
    print("       Saved to evaluation/classification_report.txt")

    # -----------------------------------------------------------------
    # 4. Confusion matrix
    # -----------------------------------------------------------------
    print("\n[4/6] Building confusion matrix heatmap...")
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="YlGnBu")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — Test Set ({model_name})")
    for i in range(n_classes):
        for j in range(n_classes):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("       Saved to evaluation/confusion_matrix.png")

    # -----------------------------------------------------------------
    # 5. Feature importance — handles both tree-based and linear models
    # -----------------------------------------------------------------
    print(f"\n[5/6] Extracting feature importance for {model_name}...")
    importance_method, top_features = extract_feature_importance(clf, vocab)
    metrics["feature_importance_method"] = importance_method
    metrics["top_20_feature_importance"] = top_features

    if top_features:
        fig, ax = plt.subplots(figsize=(9, 8))
        names = [f["symptom"] for f in reversed(top_features)]
        values = [f["importance"] for f in reversed(top_features)]
        ax.barh(names, values, color="#1F5F4E")
        xlabel = "Feature importance (Gini)" if importance_method == "gini_importance" else "Mean |coefficient|"
        ax.set_xlabel(xlabel)
        ax.set_title(f"Top 20 Most Predictive Symptoms ({model_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
        plt.close()
        print(f"       Method: {importance_method}. Saved to evaluation/feature_importance.png")
    else:
        print(f"       WARNING: {model_name} has neither feature_importances_ nor coef_ -- "
              f"no feature importance chart generated.")

    # -----------------------------------------------------------------
    # 6. Macro-average multi-class ROC-AUC (one-vs-rest)
    # -----------------------------------------------------------------
    print("\n[6/6] Computing ROC-AUC (macro, one-vs-rest)...")
    y_test_bin = label_binarize(y_test, classes=classes)
    y_proba = clf.predict_proba(X_test)
    try:
        roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        roc_auc_weighted = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
        print(f"       ROC-AUC (macro, OVR):    {roc_auc_macro:.4f}")
        print(f"       ROC-AUC (weighted, OVR): {roc_auc_weighted:.4f}")
    except ValueError as e:
        roc_auc_macro, roc_auc_weighted = None, None
        print(f"       Could not compute ROC-AUC: {e}")

    metrics["roc_auc"] = {
        "macro_ovr": round(roc_auc_macro, 4) if roc_auc_macro is not None else None,
        "weighted_ovr": round(roc_auc_weighted, 4) if roc_auc_weighted is not None else None,
    }

    metrics["dataset"] = {
        "total_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "class_names": classes,
    }

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Model:                    {model_name} ({type(clf).__name__})")
    print(f"Held-out accuracy:        {test_accuracy:.4f}")
    print(f"5-fold CV accuracy:       {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"Macro F1:                 {test_f1_macro:.4f}")
    print(f"Macro ROC-AUC:            {metrics['roc_auc']['macro_ovr']}")
    print(f"Feature importance method: {importance_method}")
    print(f"\nAll artifacts saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
