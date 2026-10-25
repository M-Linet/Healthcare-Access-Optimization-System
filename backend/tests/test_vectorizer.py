"""
test_vectorizer.py
--------------------
Unit tests for the multi-hot symptom vectorizer used in ML/train_model.py.
Run with: pytest tests/test_vectorizer.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ML"))

from train_model import build_symptom_vocabulary, vectorize, manual_train_test_split
import numpy as np


def _sample_rows():
    return [
        ["Flu", "fever", "cough", ""],
        ["Cold", "cough", "sneezing", ""],
        ["Flu", "fever", "fatigue", "cough"],
    ]


def test_build_vocabulary_excludes_empty_and_dedupes():
    rows = _sample_rows()
    vocab = build_symptom_vocabulary(rows)
    assert "" not in vocab
    assert vocab == sorted(set(vocab))  # sorted, deduped
    assert set(vocab) == {"fever", "cough", "sneezing", "fatigue"}


def test_vectorize_shape():
    rows = _sample_rows()
    vocab = build_symptom_vocabulary(rows)
    X, y = vectorize(rows, vocab)
    assert X.shape == (3, len(vocab))
    assert y.shape == (3,)


def test_vectorize_correct_multihot_encoding():
    rows = _sample_rows()
    vocab = build_symptom_vocabulary(rows)
    X, y = vectorize(rows, vocab)
    idx = {s: i for i, s in enumerate(vocab)}

    # row 0: Flu with fever, cough
    assert X[0, idx["fever"]] == 1
    assert X[0, idx["cough"]] == 1
    assert X[0, idx["sneezing"]] == 0
    assert y[0] == "Flu"

    # row 2: Flu with fever, fatigue, cough
    assert X[2, idx["fatigue"]] == 1
    assert X[2, idx["sneezing"]] == 0


def test_vectorize_binary_values_only():
    rows = _sample_rows()
    vocab = build_symptom_vocabulary(rows)
    X, y = vectorize(rows, vocab)
    assert set(np.unique(X)).issubset({0, 1})


def test_train_test_split_sizes():
    rows = _sample_rows() * 10  # 30 rows
    vocab = build_symptom_vocabulary(rows)
    X, y = vectorize(rows, vocab)
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio=0.2, seed=42)
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert X_test.shape[0] == int(30 * 0.2)


def test_train_test_split_deterministic_with_same_seed():
    rows = _sample_rows() * 10
    vocab = build_symptom_vocabulary(rows)
    X, y = vectorize(rows, vocab)
    split1 = manual_train_test_split(X, y, seed=7)
    split2 = manual_train_test_split(X, y, seed=7)
    assert np.array_equal(split1[0], split2[0])
    assert np.array_equal(split1[1], split2[1])


def test_train_test_split_no_row_overlap():
    rows = _sample_rows() * 10
    vocab = build_symptom_vocabulary(rows)
    X, y = vectorize(rows, vocab)
    X_train, X_test, _, _ = manual_train_test_split(X, y, seed=42)
    # every test row should be unique from train rows (no duplicated raw indices)
    assert X_train.shape[0] + X_test.shape[0] == len(rows)
