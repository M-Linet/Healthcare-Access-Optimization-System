# Healthcare Access Optimization System

Final year project — B.Sc. Data Science.
Student: Maina Linet Muthoni (24/00632) · Supervisor: Madam Gladys Mange

A triage system that predicts a likely condition from reported
symptoms, maps it to the minimum appropriate KEPH (Kenya Essential Package
for Health) facility level, and routes the patient to the nearest facility
capable of treating it.

## What's in here

```
healthcare-project/
├── data/
│   ├── healthcare_facilities.csv       # REAL: 10,013 facilities, all 47 counties
│   │                                    # (World Bank / energydata.info, source: opendata.go.ke)
│   ├── dataset_real_kaggle_raw.csv     # REAL: unmodified Kaggle "Disease Symptom
│   │                                    # Prediction" dataset (itachi9604), 4,920 rows
│   ├── dataset.csv                     # ACTIVE training data: 6,150 augmented cases,
│   │                                    # generated FROM the real Kaggle symptom pools
│   │                                    # (see "About the data" below — this is important)
│   ├── generate_augmented_dataset.py   # builds dataset.csv from the real Kaggle pools
│   ├── build_keph_mapping_real.py      # builds the KEPH mapping for the 41 real diseases
│   └── disease_keph_mapping.csv        # disease -> recommended min KEPH level
├── ML/
│   ├── train_model.py                  # trains the classifier; evaluate_model_comparison.py picks the best of 3
│   ├── evaluate_model.py               # full rigorous evaluation (CV, confusion matrix, etc.)
│   ├── evaluate_deduplicated.py        # leakage-diagnosis: honest CV on unique cases only
│   ├── evaluate_robustness.py          # stress-test: accuracy under noisy symptom reporting
│   ├── evaluation/                     # all evaluation charts + metrics.json (generated)
│   ├── disease_classifier.joblib       # trained model (generated)
│   ├── symptom_features.joblib         # symptom vocabulary (generated)
│   └── disease_keph_mapping.joblib     # disease -> KEPH level map (generated)
├── backend/
│   ├── main.py                         # FastAPI app (serves API + frontend)
│   ├── geo.py                          # Haversine distance + facility ranking
│   ├── red_flags.py                    # emergency symptom-combination detection
│   ├── patient_context.py              # age/sex/pregnancy/severity-aware triage adjustment
│   ├── prediction_log.py               # non-PII usage logging
│   ├── init_db.py                      # builds database.sqlite from healthcare_facilities.csv
│   ├── tests/                          # pytest unit tests (geo.py, vectorizer)
│   └── database.sqlite                 # generated: 9,736 facilities loaded into the database
├── frontend/
│   └── index.html                      # single-page web UI (symptom checker + facility finder)
└── requirements.txt
```

## IMPORTANT — about the data (read this before your defense)

This project went through three real iterations on the symptom dataset. All
three are preserved in the repo so the reasoning is auditable — this is
worth explaining to your lecturer as evidence of methodology, not hidden.

**Iteration 1 — fully synthetic (superseded).** `generate_dataset.py` built
a symptom-disease dataset from scratch using general medical knowledge,
not real data. Functional for pipeline-building, but not defensible as
"real" data if questioned.

**Iteration 2 — the real Kaggle dataset, used raw (superseded, kept for
provenance as `dataset_real_kaggle_raw.csv`).** The real, published
"Disease Symptom Prediction" dataset by itachi9604 (Kaggle) — 4,920 rows,
41 diseases, 131 real symptom tokens. Investigation found two serious
problems with using it directly:
  1. **Exact-duplicate leakage.** The file contains only **304 truly
     unique** (disease, symptom-set) cases, each copied 6–90 times to pad
     the row count. A normal random train/test split placed duplicates of
     the same case on both sides, so the model was "recognizing" cases it
     had already memorized rather than generalizing — producing a
     meaningless 100% test accuracy.
  2. **Even after fixing the leakage** (see `evaluate_deduplicated.py` —
     deduplicate to the 304 unique cases, then stratified 5-fold CV so
     every prediction is made on a genuinely unseen case), accuracy was
     **still 100%**. A robustness stress test (`evaluate_robustness.py`,
     randomly dropping/adding symptoms at test time up to 50% corruption)
     confirmed this wasn't a fluke: accuracy stayed above 99% even under
     heavy noise. The real cause: this dataset assigns each disease a
     small number of fixed, essentially non-overlapping symptom templates,
     making it close to a lookup table rather than a realistic diagnostic
     problem. This is a genuine, verifiable property of the dataset, not
     a bug in the evaluation.

**Iteration 3 — real-data-grounded augmentation (what the system actually
trains on now, `dataset.csv`).** `generate_augmented_dataset.py` keeps the
**real Kaggle disease-symptom associations as ground truth** — every
symptom pool per disease is the union of symptoms actually recorded for
that disease in the real dataset, nothing invented — and generates a
larger set of realistic patient *cases* by sampling from those real pools
with:
  - non-specific symptom overlap across diseases (fatigue, fever, headache
    etc. — all real Kaggle symptom tokens, chosen for realistic cross-
    disease prevalence)
  - a chance of borrowing a symptom from a different disease's real pool
    (simulates overlapping/comorbid presentations)
  - simulated incomplete reporting (a symptom randomly dropped)

This does not alter or fabricate medical knowledge — it addresses the
fact that 304 unique, idealized textbook cases is both too small and too
clean to train or evaluate a model meant to reflect real diagnostic
uncertainty. This is a standard, defensible data augmentation technique,
directly analogous to image augmentation (rotate/crop/noise) in computer
vision — the label/ground-truth relationship is preserved, only the
surface-level presentation varies.

**Resulting honest metrics** (see `ML/evaluation/metrics.json`):
  - Deployed model: Logistic Regression (best of three compared by 5-fold CV)
  - Held-out test accuracy: **92.44%**
  - 5-fold cross-validation: **92.37% ± 0.64%** (folds:
    91.30% / 92.85% / 92.36% / 92.20% / 93.17%)
  - Train accuracy: 95.22% (a 2.78-point train/test gap)
  - Compared models (CV accuracy): Logistic Regression 92.37%,
    Gradient Boosting 89.33%, Random Forest 89.28%
  - Confusion matrix shows realistic, clinically-sensible confusions
    concentrated among genuinely similar conditions (Hepatitis B/C/D/E
    confusing with each other and with Jaundice, Asthma variants
    overlapping) — not random noise.

- **`healthcare_facilities.csv` is real and unmodified.** This is the
  World Bank / energydata.info "Kenya — Healthcare Facilities" dataset,
  sourced from Kenya's open data portal (opendata.go.ke), covering
  **10,013 facilities across all 47 counties**. `init_db.py` maps each
  facility's raw `Type` field (Dispensary, Health Centre, District
  Hospital, etc.) to a KEPH level 2–6 using Kenya's Essential Package for
  Health definitions — see `TYPE_TO_KEPH` in `backend/init_db.py`. 279
  rows are excluded from routing because they're ancillary (stand-alone
  labs, VCT centres, admin offices, a funeral home) rather than places to
  send a walk-in patient. The resulting distribution is realistic for
  Kenya's health pyramid: 7,865 dispensary-level, 1,331 health centres,
  523 hospitals, 12 provincial/county referral, 3 national referral
  (Kenyatta National, Moi Teaching & Referral, National Spinal Injury
  Hospital).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run it (three steps, in order)

```bash
# 1. Generate the training data (only needed once, or after editing
#    generate_augmented_dataset.py — dataset.csv is already included, so
#    you can skip this step and go straight to step 2)
cd data
python generate_augmented_dataset.py

# 2. Train the model
cd ../ML
python train_model.py

# Optional but recommended: run the full evaluation suite (cross-validation,
# confusion matrix, feature importance — everything in ML/evaluation/)
python evaluate_model.py

# 3. Build the facility database and start the server
cd ../backend
python init_db.py
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser — the whole app (symptom
checker, KEPH ladder, and nearest-facility finder) is served from that one
link. No separate frontend build step, no Android Studio needed.

## API reference

| Method | Endpoint                  | Purpose                                          |
|--------|----------------------------|--------------------------------------------------|
| GET    | `/api/symptoms`            | List all known symptom tokens                    |
| POST   | `/api/predict`              | `{symptoms: [...]}` → predicted disease + KEPH level |
| POST   | `/api/nearest-facilities`   | `{latitude, longitude, min_keph_level?, limit?}` → ranked facilities |
| POST   | `/api/triage`               | `{symptoms, latitude, longitude}` → prediction + matching facilities combined |
| GET    | `/api/facilities`           | Full facility list                               |
| GET    | `/health`                   | Healthcheck                                      |

## Retraining after any data change

1. Edit `data/generate_augmented_dataset.py` if you want different noise
   parameters, or replace `data/dataset_real_kaggle_raw.csv` with an updated
   real dataset.
2. Re-run `python data/generate_augmented_dataset.py` to rebuild `dataset.csv`.
3. Re-run `python ML/train_model.py` and, ideally, `python ML/evaluate_model.py`.
4. Restart the server — no other changes needed.

## Facility data notes / known limitations

- KEPH level is *inferred* from the `Type` column, since the source dataset
  doesn't carry an official KEPH tag. This is a reasonable approximation but
  not authoritative — cross-check against the real KMHFR registry
  (`kmhfr.health.go.ke`) if this needs to be defensible for a live deployment.
- 84 facilities show `(0,0)`-style or missing coordinates in the raw file
  and are dropped automatically; ~200 more are ancillary types (labs, VCT
  centres) and excluded from routing by design (see `EXCLUDED_TYPES` in
  `init_db.py`) — adjust that set if you want them included.
- Private facilities (clinics, nursing homes) are included alongside public
  ones. If your project scope is public-sector-only, filter on `ownership`
  in `init_db.py`'s `transform_row()` before inserting.

## Deploying so it's reachable from a phone / off your laptop

Right now `uvicorn` serves locally. For a real deployed link, the simplest
options are:
- **Render / Railway / Fly.io** (free tiers): push this repo, point them at
  `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
- **ngrok**: run `ngrok http 8000` while `uvicorn` is running locally, for a
  temporary public link — good for demos to your supervisor.

## Next steps (from your original plan)

1. FastAPI inference endpoints — done (`/api/predict`)
2. Haversine distance routing — done (`backend/geo.py`, `/api/nearest-facilities`)
3. Web client — done (`frontend/index.html`, replaces the Android Studio plan)
4. Real facility dataset (energydata.info, 9,736 facilities loaded into the database) — done
5. Emergency red-flag detection — done (`backend/red_flags.py`)
6. Patient context (age/sex/pregnancy) & symptom severity — done (`backend/patient_context.py`)
7. Confidence floor for uncertain predictions — done
8. Swahili/English language toggle in the UI — done
9. Unit tests for `geo.py` and the vectorizer — done (`backend/tests/`)
10. Rate limiting on prediction endpoints — done (slowapi, 30/min)
11. Non-PII prediction logging + usage stats endpoint — done (`/api/stats`)
12. Real Kaggle symptom dataset — done, with full leakage/robustness
    diagnosis and a defensible augmented-training approach (see "About the
    data" above) — deployed Logistic Regression: test accuracy 92.44%, 5-fold CV 92.37% ± 0.64%
13. Consider adding authentication / patient session history if scope grows
14. Road-distance routing (OSRM) with straight-line fallback — done
15. Multi-model comparison (Logistic Regression, Random Forest, Gradient
    Boosting) — done; a paired statistical test is still future work
16. Consider SHAP/permutation importance for more rigorous interpretability
    than Gini-based feature importance alone

## What's new in this version (realism pass)

Building on the working prototype, this version adds:

- **Emergency red-flag detection** (`backend/red_flags.py`) — a rule-based layer
  that runs independently of the ML model. Dangerous symptom combinations
  (e.g. chest pain + breathlessness + sweating, slurred speech + limb
  weakness) trigger an `is_emergency: true` flag, specific advice, and
  Kenya's emergency number (999/112), and force routing to at least a
  County Referral Hospital (KEPH L5) regardless of what the classifier
  predicted.
- **Confidence floor** — predictions under 40% confidence are marked
  `low_confidence: true`. The frontend shows a caution notice instead of
  asserting a diagnosis with false certainty.
- **Patient context** (`backend/patient_context.py`) — optional age, sex,
  and pregnancy fields adjust the recommended KEPH level: pregnancy with
  relevant symptoms routes to at least a Health Centre for maternity
  capability; infants and patients 65+ get a precautionary bump.
- **Per-symptom severity** — users can mark each symptom mild/moderate/severe;
  two or more "severe" symptoms bump the recommended KEPH level by one tier.
- **Swahili/English toggle** — the whole UI (labels, KEPH ladder, notices)
  switches language client-side with no reload.
- **Rate limiting** — `/api/predict` and `/api/triage` are capped at
  30 requests/minute per IP (via `slowapi`) to prevent abuse.
- **Non-PII usage logging** (`backend/prediction_log.py`) — every prediction
  logs disease, confidence, KEPH level, and whether a red flag fired, to
  `backend/prediction_log.sqlite`. Coordinates are rounded to ~11km precision
  before storage — nowhere near enough to identify a specific person's
  location. Query `/api/stats` for aggregate numbers (useful for your defense).
- **Unit tests** (`backend/tests/`) — 15 tests covering Haversine distance
  correctness/symmetry, facility ranking and KEPH filtering, the symptom
  vectorizer's multi-hot encoding, and the train/test split's determinism.
  Run with `pytest tests/ -v` from the `backend/` folder.

### New API fields worth knowing about

`/api/predict` and the `prediction` object inside `/api/triage` now return:

```json
{
  "predicted_disease": "...",
  "confidence": 0.79,
  "low_confidence": false,
  "recommended_keph_level": 6,
  "base_keph_level": 6,
  "context_adjustments": ["..."],
  "red_flags": [{"label": "...", "advice": "...", "matched_symptoms": [...]}],
  "is_emergency": true,
  "emergency_number": "999 / 112",
  "top_alternatives": [...]
}
```

Optional request fields: `severity` (dict of symptom→mild/moderate/severe),
`age` (int), `sex` (male/female/other), `pregnant` (bool).


- `notebook/Data_Exploration_and_Analysis.ipynb` is the data story: what the symptom data really contains, how Kenya's facilities compare with what the conditions need, how far the nearest referral hospital is, and where the model fails. The charts are also saved in `ML/evaluation/story/`.
- `ML/extended_search/` holds a small manual search over model settings and model types. See the README in that folder.

## Running the tests

```
cd backend
python -m pytest tests -v
```

49 automated tests (distance, symptom encoding, red-flag rules, care level rules, routing fallback and the API).
