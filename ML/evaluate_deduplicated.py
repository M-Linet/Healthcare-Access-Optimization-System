"""
evaluate_deduplicated.py
---------------------------
The real Kaggle "Disease Symptom Prediction" dataset (data/dataset.csv, 4,920
rows) turns out to contain only 304 truly distinct (disease, symptom-set)
cases -- the rest are exact duplicate rows padding the file out (confirmed:
average 16.2x duplication per case). A plain random or even group-aware
80/20 split still leaves some rare diseases entirely out of the test set,
because with only 5-10 unique presentations per disease across 41 classes,
group-level splitting is too coarse.

This script evaluates honestly on the deduplicated 304-row dataset using
STRATIFIED 5-fold cross-validation, so:
  - No duplicate row of the same case can ever leak between train and test
    (there are no duplicates left at all -- each row is unique)
  - Every disease is guaranteed to appear in every fold's test set
    (stratification), since every class has >= 5 unique examples

This is the number that should actually be quoted as "real-world" model
accuracy for this dataset -- not the 100% figure produced by evaluating on
data containing exact duplicates of the training set.

Usage:
    python evaluate_deduplicated.py
"""

import os
import csv
import json
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_model import load_raw_rows, build_symptom_vocabulary, DATA_PATH, RANDOM_SEED

OUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)


def deduplicate_rows(rows):
    """Collapse to one row per distinct (disease, symptom-set) case."""
    seen = {}
    for row in rows:
        disease = row[0].strip()
        symptoms = frozenset(c.strip() for c in row[1:] if c.strip())
        key = (disease, symptoms)
        if key not in seen:
            seen[key] = row
    return list(seen.values())


def vectorize_dedup(rows, vocab):
    index_of = {s: i for i, s in enumerate(vocab)}
    X = np.zeros((len(rows), len(vocab)), dtype=np.uint8)
    y = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        y[i] = row[0].strip()
        for cell in row[1:]:
            c = cell.strip()
            if c and c in index_of:
                X[i, index_of[c]] = 1
    return X, y


def main():
    print("=" * 70)
    print("DEDUPLICATED EVALUATION — honest accuracy on the real Kaggle dataset")
    print("=" * 70)

    header, all_rows = load_raw_rows(DATA_PATH)
    vocab = build_symptom_vocabulary(all_rows)

    dedup_rows = deduplicate_rows(all_rows)
    print(f"\nOriginal rows: {len(all_rows)}")
    print(f"Deduplicated unique cases: {len(dedup_rows)}")
    print(f"Duplication factor: {len(all_rows)/len(dedup_rows):.1f}x")

    X, y = vectorize_dedup(dedup_rows, vocab)
    classes = sorted(set(y))
    print(f"\nDeduplicated dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(classes)} classes")

    # Confirm every class has enough samples for 5-fold stratification
    from collections import Counter
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    n_folds = min(5, min_class_count)
    print(f"Smallest class has {min_class_count} unique examples -> using {n_folds}-fold stratified CV")

    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    print(f"\nRunning {n_folds}-fold stratified cross-validation on {len(dedup_rows)} unique cases...")
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"Fold scores: {[round(s, 4) for s in cv_scores]}")
    print(f"Mean accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Out-of-fold predictions for a proper classification report / confusion matrix
    y_pred_oof = cross_val_predict(clf, X, y, cv=skf, n_jobs=-1)
    acc_oof = accuracy_score(y, y_pred_oof)
    f1_oof = f1_score(y, y_pred_oof, average="macro", zero_division=0)
    print(f"\nOut-of-fold accuracy (every prediction made on a held-out fold): {acc_oof:.4f}")
    print(f"Out-of-fold macro F1: {f1_oof:.4f}")

    report = classification_report(y, y_pred_oof, zero_division=0)
    print("\nOut-of-fold classification report (every disease evaluated on unseen data only):")
    print(report)

    with open(os.path.join(OUT_DIR, "deduplicated_classification_report.txt"), "w") as f:
        f.write(f"Deduplicated evaluation: {len(dedup_rows)} unique cases (from {len(all_rows)} total rows)\n")
        f.write(f"{n_folds}-fold stratified CV, out-of-fold predictions\n\n")
        f.write(report)

    # Confusion matrix on deduplicated out-of-fold predictions
    cm = confusion_matrix(y, y_pred_oof, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("True disease")
    ax.set_title(f"Deduplicated Confusion Matrix — {n_folds}-fold CV, out-of-fold predictions\n"
                 f"({len(dedup_rows)} unique cases, no duplicate leakage)")
    fig.colorbar(im, ax=ax, label="Proportion of true class")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "deduplicated_confusion_matrix.png"), dpi=150)
    plt.close()

    metrics = {
        "original_row_count": len(all_rows),
        "unique_case_count": len(dedup_rows),
        "duplication_factor": round(len(all_rows) / len(dedup_rows), 2),
        "n_classes": len(classes),
        "min_examples_per_class": min_class_count,
        "n_folds_used": n_folds,
        "cv_fold_scores": [round(s, 4) for s in cv_scores],
        "cv_mean_accuracy": round(cv_scores.mean(), 4),
        "cv_std_accuracy": round(cv_scores.std(), 4),
        "out_of_fold_accuracy": round(acc_oof, 4),
        "out_of_fold_macro_f1": round(f1_oof, 4),
        "note": (
            "This is the honest accuracy estimate for the real Kaggle dataset. "
            "The naive 100% figure from a plain 80/20 split is inflated by data "
            "leakage: the source file duplicates each of only 304 unique cases "
            "up to 90 times, so a random split places copies of the same case "
            "in both train and test. This evaluation deduplicates first, then "
            "uses stratified k-fold cross-validation so every prediction is "
            "made on a truly unseen case."
        ),
    }
    with open(os.path.join(OUT_DIR, "deduplicated_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print(f"HONEST accuracy estimate: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print(f"(compare to the misleading 100% from evaluating on duplicate-contaminated data)")
    print("=" * 70)
    print(f"\nSaved to {OUT_DIR}/deduplicated_metrics.json, deduplicated_confusion_matrix.png, "
          f"deduplicated_classification_report.txt")


if __name__ == "__main__":
    main()
