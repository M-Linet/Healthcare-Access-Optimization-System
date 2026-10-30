"""
confidence_calibration.py
How often is the model right at each confidence level? Used to choose
CONFIDENCE_FLOOR in backend/main.py.

Two views, both on the group-aware held-out split used by train_model.py:
  1. the held-out rows as they are
  2. the same cases with only 2 or 3 of their symptoms kept, which is closer
     to what a real user types

Run from the project root:  python ML/evaluation/confidence_calibration.py
Writes ML/evaluation/confidence_calibration.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))
import train_model as t  # noqa: E402

BINS = [(0, .25), (.25, .4), (.4, .5), (.5, .6), (.6, .75), (.75, .9), (.9, 1.01)]
FLOORS = [0.4, 0.5, 0.6, 0.7]


def bucket_table(conf, ok):
    rows = []
    for lo, hi in BINS:
        k = (conf >= lo) & (conf < hi)
        if k.sum():
            rows.append({"from": lo, "to": min(hi, 1.0), "n": int(k.sum()),
                         "share_correct": round(float(ok[k].mean()), 3)})
    return rows


def main():
    _, rows = t.load_raw_rows(t.DATA_PATH)
    vocab = t.build_symptom_vocabulary(rows)
    X, y = t.vectorize(rows, vocab)
    groups = t.build_case_groups(rows)
    Xtr, Xte, ytr, yte = t.group_train_test_split(X, y, groups)
    model = LogisticRegression(max_iter=1000, random_state=t.RANDOM_SEED).fit(Xtr, ytr)

    def score(Xe, ye):
        P = model.predict_proba(Xe)
        return P.max(1), model.classes_[P.argmax(1)] == ye

    conf, ok = score(Xte, yte)
    out = {"held_out_as_is": {"n": int(len(yte)), "accuracy": round(float(ok.mean()), 3),
                              "by_confidence": bucket_table(conf, ok)}}

    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for xi, yi in zip(Xte, yte):
        idx = np.flatnonzero(xi)
        if len(idx) < 3:
            continue
        for k in (2, 3):
            v = np.zeros_like(xi)
            v[rng.choice(idx, size=k, replace=False)] = 1
            Xs.append(v)
            ys.append(yi)
    conf, ok = score(np.array(Xs), np.array(ys))
    out["partial_2_or_3_symptoms"] = {
        "n": int(len(ys)), "accuracy": round(float(ok.mean()), 3),
        "by_confidence": bucket_table(conf, ok),
        "by_floor": [{"floor": f, "share_marked_confident": round(float((conf >= f).mean()), 3),
                      "accuracy_when_confident": round(float(ok[conf >= f].mean()), 3),
                      "accuracy_when_flagged": round(float(ok[conf < f].mean()), 3)} for f in FLOORS],
    }
    path = Path(__file__).resolve().parent / "confidence_calibration.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
