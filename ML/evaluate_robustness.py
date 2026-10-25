"""
evaluate_robustness.py
-------------------------
Both the real Kaggle dataset and cross-validation confirm the model gets
100% accuracy on clean, exactly-as-trained symptom presentations. That's a
genuine finding, not a bug -- but it's also not a realistic estimate of how
the system will perform in the field, because real patients don't report
symptoms as cleanly as a fixed dataset template. They forget symptoms,
report irrelevant ones, and describe things imprecisely.

This script does NOT alter the dataset or ground-truth labels. It takes the
model exactly as trained, and perturbs only the TEST-time input to simulate
imperfect real-world reporting:
    - randomly dropping a reported symptom (patient forgets to mention it)
    - randomly adding an unrelated symptom (patient over-reports, or has a
      mild unrelated complaint)

This is standard robustness / stress-testing methodology: measuring how
much accuracy degrades as input quality degrades, at several noise levels.
The result is a realistic operating range for expected field accuracy,
not a single potentially-misleading number.

Usage:
    python evaluate_robustness.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from train_model import load_raw_rows, build_symptom_vocabulary, DATA_PATH, RANDOM_SEED
from evaluate_deduplicated import deduplicate_rows, vectorize_dedup

OUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

NOISE_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
N_TRIALS = 20  # repeat each noise level this many times with different random perturbations


def perturb(X_row, noise_prob, rng, n_features):
    """Randomly drop a present symptom and/or add an absent one, each with noise_prob."""
    row = X_row.copy()
    present = np.where(row == 1)[0]
    absent = np.where(row == 0)[0]

    if len(present) > 1 and rng.random() < noise_prob:
        drop_idx = rng.choice(present)
        row[drop_idx] = 0

    if len(absent) > 0 and rng.random() < noise_prob:
        add_idx = rng.choice(absent)
        row[add_idx] = 1

    return row


def main():
    print("=" * 70)
    print("ROBUSTNESS EVALUATION — accuracy under simulated imperfect symptom reporting")
    print("=" * 70)

    header, all_rows = load_raw_rows(DATA_PATH)
    vocab = build_symptom_vocabulary(all_rows)
    dedup_rows = deduplicate_rows(all_rows)
    X, y = vectorize_dedup(dedup_rows, vocab)
    n_features = X.shape[1]

    print(f"\nUsing {X.shape[0]} unique cases, {n_features} symptom features, {len(set(y))} diseases")

    # Fixed 70/30 stratified-ish split (simple deterministic split since classes are tiny)
    rng_split = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(X.shape[0])
    rng_split.shuffle(idx)
    n_test = int(0.3 * len(idx))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)

    clean_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\nClean (unperturbed) test accuracy: {clean_acc:.4f}")

    results = []
    for noise in NOISE_LEVELS:
        trial_accs = []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(RANDOM_SEED + trial)
            X_noisy = np.array([perturb(row, noise, rng, n_features) for row in X_test])
            preds = clf.predict(X_noisy)
            trial_accs.append(accuracy_score(y_test, preds))
        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs)
        results.append({"noise_level": noise, "mean_accuracy": round(mean_acc, 4), "std_accuracy": round(std_acc, 4)})
        print(f"  noise={noise:.0%}  ->  accuracy = {mean_acc:.4f} (+/- {std_acc:.4f}, n={N_TRIALS} trials)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    noise_vals = [r["noise_level"] for r in results]
    means = [r["mean_accuracy"] for r in results]
    stds = [r["std_accuracy"] for r in results]
    ax.errorbar([n * 100 for n in noise_vals], means, yerr=stds, marker='o',
                color="#1F5F4E", capsize=4, linewidth=2)
    ax.set_xlabel("Simulated reporting noise (% chance of dropped/extra symptom)")
    ax.set_ylabel("Model accuracy")
    ax.set_title("Accuracy vs. Simulated Imperfect Symptom Reporting")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "robustness_curve.png"), dpi=150)
    plt.close()

    metrics = {
        "clean_test_accuracy": round(clean_acc, 4),
        "noise_sweep": results,
        "n_trials_per_level": N_TRIALS,
        "note": (
            "Ground truth labels and training data are untouched. Only test-time "
            "input is perturbed (symptom dropped and/or an unrelated symptom added, "
            "each with the given probability) to simulate imperfect real-world "
            "patient symptom reporting. This estimates expected field accuracy "
            "under realistic reporting conditions, distinct from clean-data accuracy."
        ),
    }
    with open(os.path.join(OUT_DIR, "robustness_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Clean accuracy: {clean_acc:.2%}")
    print("Under realistic reporting noise (20-30% symptom drop/add rate),")
    mid = [r for r in results if r["noise_level"] in (0.20, 0.30)]
    for r in mid:
        print(f"  at {r['noise_level']:.0%} noise: {r['mean_accuracy']:.2%} accuracy")
    print("=" * 70)
    print(f"\nSaved to {OUT_DIR}/robustness_metrics.json, robustness_curve.png")


if __name__ == "__main__":
    main()
