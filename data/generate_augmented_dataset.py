"""
generate_augmented_dataset.py
--------------------------------
The real Kaggle dataset (data/dataset.csv) contains only 304 truly unique
disease-symptom presentations (41 diseases x 5-10 fixed templates each),
padded to 4,920 rows via exact duplication. Verified independently (see
ML/evaluate_deduplicated.py, ML/evaluate_robustness.py): even with
duplicate-leakage removed and the model stress-tested against corrupted
input, accuracy stays at 99-100%, because the underlying 304 cases use
small, nearly non-overlapping symptom templates per disease.

That's not a usable evaluation set for a system meant to reflect real
diagnostic uncertainty -- in practice, the same symptoms (fever, fatigue,
headache, nausea...) show up across many different diseases, and patients
report incompletely.

This script keeps the real medical knowledge -- which symptoms are
associated with which disease -- taken directly from the real Kaggle data
(the per-disease symptom POOL is the union of every symptom ever recorded
for that disease across its unique presentations), and generates a larger,
more realistic set of synthetic PATIENT CASES by sampling from those real
pools with:
    - non-specific/common symptom overlap across diseases (real patients
      with different conditions often share fever, fatigue, headache, etc.)
    - occasional symptoms borrowed from a different disease's real pool
      (comorbid or overlapping presentations)
    - simulated incomplete reporting (a symptom is dropped)

This does NOT alter or fabricate the medical ground truth (which symptoms
belong to which disease is taken directly from the real dataset). It
generates a properly-sized, realistically-noisy set of CASES for training
and evaluation, which the raw 304-case file is too small and too clean to
provide on its own.

Usage:
    python generate_augmented_dataset.py
"""

import csv
import os
import random
from collections import defaultdict

random.seed(42)

SOURCE_CSV = os.path.join(os.path.dirname(__file__), "dataset.csv")
OUT_CSV = os.path.join(os.path.dirname(__file__), "dataset_augmented.csv")
OUT_KEPH = os.path.join(os.path.dirname(__file__), "disease_keph_mapping.csv")  # unchanged, reused as-is

# Symptoms that are genuinely non-specific across many real illnesses.
# All of these are real tokens that already exist in the Kaggle symptom
# vocabulary -- nothing invented here, just chosen for their real-world
# cross-disease prevalence.
COMMON_NONSPECIFIC = [
    "fatigue", "headache", "high_fever", "mild_fever", "nausea", "chills",
    "malaise", "vomiting", "sweating", "loss_of_appetite", "dizziness",
    "weakness_in_limbs", "lethargy",
]


def load_real_pools(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and r[0].strip()]

    pools = defaultdict(set)
    all_symptoms = set()
    for r in rows:
        disease = r[0].strip()
        for cell in r[1:]:
            c = cell.strip()
            if c:
                pools[disease].add(c)
                all_symptoms.add(c)
    return pools, sorted(all_symptoms)


def generate_augmented_rows(
    pools,
    all_symptoms,
    n_per_disease=150,
    common_symptom_prob=0.55,
    cross_disease_prob=0.30,
    drop_prob=0.20,
    min_symptoms=2,
):
    diseases = list(pools.keys())
    common_available = [s for s in COMMON_NONSPECIFIC if s in all_symptoms]

    rows = []
    for disease in diseases:
        pool = list(pools[disease])
        for _ in range(n_per_disease):
            k = random.randint(min(min_symptoms, len(pool)), len(pool))
            chosen = set(random.sample(pool, k))

            if common_available and random.random() < common_symptom_prob:
                chosen.add(random.choice(common_available))

            if random.random() < cross_disease_prob:
                other_disease = random.choice([d for d in diseases if d != disease])
                other_pool = list(pools[other_disease])
                if other_pool:
                    chosen.add(random.choice(other_pool))

            if len(chosen) > min_symptoms and random.random() < drop_prob:
                chosen.discard(random.choice(list(chosen)))

            rows.append([disease] + list(chosen))

    random.shuffle(rows)
    return rows


def write_csv(rows, path):
    max_symptoms = max(len(r) - 1 for r in rows)
    header = ["Disease"] + [f"Symptom_{i+1}" for i in range(max_symptoms)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            padded = r + [""] * (max_symptoms - (len(r) - 1))
            writer.writerow(padded)


if __name__ == "__main__":
    pools, all_symptoms = load_real_pools(SOURCE_CSV)
    print(f"Loaded real symptom pools for {len(pools)} diseases from {SOURCE_CSV}")
    print(f"Real symptom vocabulary: {len(all_symptoms)} tokens")

    rows = generate_augmented_rows(
        pools, all_symptoms, n_per_disease=150,
        common_symptom_prob=0.75, cross_disease_prob=0.50, drop_prob=0.30,
    )
    write_csv(rows, OUT_CSV)
    print(f"Generated {len(rows)} augmented cases -> {OUT_CSV}")
