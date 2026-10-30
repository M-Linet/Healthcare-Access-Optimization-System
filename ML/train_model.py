"""
train_model.py
----------------
Trains the disease classifier used by the Healthcare Access Optimization System.

Pipeline:
  1. Load data/dataset.csv (Disease, Symptom_1..Symptom_N)
  2. Build a pure Python/NumPy multi-hot symptom vectorizer (no pandas get_dummies,
     no pyarrow-backed string ops — avoids Python 3.14 / pyarrow indexing issues).
  3. Manual NumPy shuffle + split (train/test), no sklearn train_test_split, so
     behavior is identical across platforms/versions.
  4. Train a LogisticRegression (the model the app uses; chosen over Random
     Forest and Gradient Boosting by 5-fold CV, see evaluate_model_comparison.py).
  5. Evaluate on the held-out, group-aware test split, then refit on all the
     data for the saved model (this is how the deployed model was made).
  6. Save the model, the symptom feature list (for building inference vectors),
     and the disease->KEPH level map to ML/ as .joblib artifacts.

Usage:
    python train_model.py
"""

import csv
import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
KEPH_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "disease_keph_mapping.csv")
OUT_DIR = os.path.dirname(__file__)

RANDOM_SEED = 42


def load_raw_rows(path):
    """Read the CSV manually (avoids pandas/pyarrow parsing quirks)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row and row[0].strip()]
    return header, rows


def build_symptom_vocabulary(rows):
    """Collect the full sorted set of distinct symptom tokens across all rows."""
    vocab = set()
    for row in rows:
        for cell in row[1:]:
            cell = cell.strip()
            if cell:
                vocab.add(cell)
    return sorted(vocab)


def vectorize(rows, vocab):
    """
    Pure NumPy multi-hot vectorizer.
    Returns:
        X: (n_samples, n_features) binary matrix
        y: (n_samples,) array of disease labels (strings)
    """
    index_of = {symptom: i for i, symptom in enumerate(vocab)}
    n_samples = len(rows)
    n_features = len(vocab)
    X = np.zeros((n_samples, n_features), dtype=np.uint8)
    y = np.empty(n_samples, dtype=object)

    for i, row in enumerate(rows):
        disease = row[0].strip()
        y[i] = disease
        for cell in row[1:]:
            cell = cell.strip()
            if cell and cell in index_of:
                X[i, index_of[cell]] = 1

    return X, y


def manual_train_test_split(X, y, test_ratio=0.2, seed=RANDOM_SEED):
    """Explicit NumPy shuffling + index-based split (cross-platform deterministic)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = int(n * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def build_case_groups(rows):
    """
    Assigns a group ID to each row based on its exact (disease, symptom-set)
    combination. Rows with the same group ID are EXACT duplicates of the same
    underlying case.

    This matters because several public symptom-disease datasets (including
    the real Kaggle "Disease Symptom Prediction" dataset) pad their row count
    by repeating a small number of distinct cases dozens of times. A plain
    random train/test split will place duplicates of the same case on both
    sides of the split -- the model then "generalizes" by recognizing an
    exact case it already memorized, not by learning the underlying pattern.
    That produces inflated, meaningless accuracy (in practice, close to 100%).

    Grouping by exact case and splitting at the GROUP level (see
    group_train_test_split below) closes that leak: every duplicate of a
    case stays entirely in train or entirely in test, never split across both.
    """
    groups = []
    seen = {}
    for row in rows:
        disease = row[0].strip()
        symptoms = frozenset(c.strip() for c in row[1:] if c.strip())
        key = (disease, symptoms)
        if key not in seen:
            seen[key] = len(seen)
        groups.append(seen[key])
    return np.array(groups)


def group_train_test_split(X, y, groups, test_ratio=0.2, seed=RANDOM_SEED):
    """
    Like manual_train_test_split, but splits by case GROUP rather than by row,
    so duplicate rows of the same case can never appear on both sides.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    n_test_groups = max(1, int(len(unique_groups) * test_ratio))
    test_groups = set(unique_groups[:n_test_groups])

    test_mask = np.array([g in test_groups for g in groups])
    train_mask = ~test_mask

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def load_keph_mapping(path):
    mapping = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if row:
                mapping[row[0].strip()] = int(row[1])
    return mapping


def main():
    print("Loading dataset...")
    header, rows = load_raw_rows(DATA_PATH)
    print(f"  {len(rows)} rows loaded, {len(header) - 1} symptom columns in schema")

    print("Building symptom vocabulary...")
    vocab = build_symptom_vocabulary(rows)
    print(f"  {len(vocab)} distinct symptoms")

    print("Checking for duplicate cases (same disease + identical symptom set)...")
    groups = build_case_groups(rows)
    n_unique_cases = len(np.unique(groups))
    if n_unique_cases < len(rows):
        dup_factor = len(rows) / n_unique_cases
        print(f"  WARNING: {len(rows)} rows contain only {n_unique_cases} distinct cases "
              f"(avg {dup_factor:.1f}x duplication).")
        print(f"  Using GROUP-AWARE train/test split so duplicate rows of the same case")
        print(f"  cannot appear in both train and test (prevents data leakage / inflated accuracy).")
    else:
        print(f"  No duplication detected ({n_unique_cases} unique cases).")

    print("Vectorizing (multi-hot encoding)...")
    X, y = vectorize(rows, vocab)
    print(f"  Feature matrix shape: {X.shape}")

    print("Splitting train/test (80/20, group-aware, manual NumPy shuffle)...")
    X_train, X_test, y_train, y_test = group_train_test_split(X, y, groups)
    print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    print("Training LogisticRegression on the train split (for evaluation only)...")
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    print("Evaluating...")
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  Test accuracy (leakage-free): {acc * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, preds, zero_division=0))

    print("Refitting on all rows for the saved model (as the deployed model was)...")
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X, y)

    print("Loading disease -> KEPH level mapping...")
    keph_mapping = load_keph_mapping(KEPH_MAP_PATH)

    print("Saving artifacts to ML/ ...")
    joblib.dump(clf, os.path.join(OUT_DIR, "disease_classifier.joblib"))
    joblib.dump(vocab, os.path.join(OUT_DIR, "symptom_features.joblib"))
    joblib.dump(keph_mapping, os.path.join(OUT_DIR, "disease_keph_mapping.joblib"))

    print("Done.")
    print(f"  - disease_classifier.joblib")
    print(f"  - symptom_features.joblib ({len(vocab)} features)")
    print(f"  - disease_keph_mapping.joblib ({len(keph_mapping)} diseases)")


if __name__ == "__main__":
    main()
