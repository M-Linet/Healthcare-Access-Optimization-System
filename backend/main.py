"""
main.py
--------
FastAPI backend for the Healthcare Access Optimization System.

Endpoints:
    GET  /                       -> serves the web frontend (index.html)
    GET  /api/symptoms           -> list of all known symptom tokens (for building the UI form)
    POST /api/predict            -> symptom (+ optional patient context) -> predicted disease,
                                     confidence, KEPH level, red-flag warnings
    POST /api/nearest-facilities -> {latitude, longitude, min_keph_level?, limit?} -> ranked facility list
    POST /api/triage              -> full flow: prediction + red flags + nearest facilities, in one call
    GET  /api/facilities         -> full facility list (for map display)
    GET  /api/facility-stats     -> aggregate facility registry statistics
    GET  /api/stats              -> non-PII usage stats (predictions logged, top diseases, etc.)
    GET  /api/model-performance  -> saved model evaluation report
    GET  /api/model-comparison   -> multi-model comparison report
    GET  /health                 -> simple healthcheck

Run:
    uvicorn main:app --reload --port 8000
Then open: http://localhost:8000
"""

import os
import json
import logging
import sqlite3

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from geo import rank_facilities_by_distance
from red_flags import check_red_flags, EMERGENCY_NUMBER_KE
from patient_context import adjust_keph_level
from road_distance import rank_by_road_distance
import prediction_log
import security

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(__file__)
ML_DIR = os.path.join(BASE_DIR, "..", "ML")
DB_PATH = os.path.join(BASE_DIR, "database.sqlite")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
EVAL_DIR = os.path.join(ML_DIR, "evaluation")

# Below this model confidence, we don't assert a diagnosis — we say "unclear, see a clinician"
CONFIDENCE_FLOOR = 0.40

# ---------------------------------------------------------------------------
# App setup — rate limiting, CORS, and security middleware
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Healthcare Access Optimization System")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=security.get_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Security hardening — these were previously imported but never activated.
# Order: GZip first (compresses large JSON like /api/facilities), then
# size limiting and timing, then security headers on every response, then
# the safe exception handler last so it wraps everything below it.
app.add_middleware(GZipMiddleware, minimum_size=1000)
security.add_request_size_limit(app, max_bytes=100_000)
security.add_request_timing(app)
security.add_security_headers(app)
security.add_safe_exception_handler(app)

# ---------------------------------------------------------------------------
# Load ML artifacts at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(ML_DIR, "disease_classifier.joblib")
FEATURES_PATH = os.path.join(ML_DIR, "symptom_features.joblib")
KEPH_MAP_PATH = os.path.join(ML_DIR, "disease_keph_mapping.joblib")

if not all(os.path.exists(p) for p in [MODEL_PATH, FEATURES_PATH, KEPH_MAP_PATH]):
    raise RuntimeError(
        "ML artifacts not found. Run `python ML/train_model.py` first."
    )

classifier = joblib.load(MODEL_PATH)
symptom_vocab = joblib.load(FEATURES_PATH)  # sorted list of symptom strings
disease_keph_map = joblib.load(KEPH_MAP_PATH)  # dict: disease -> keph level
symptom_index = {s: i for i, s in enumerate(symptom_vocab)}

VALID_SEVERITIES = {"mild", "moderate", "severe"}
VALID_SEX = {"male", "female", "other"}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    symptoms: List[str] = Field(..., description="Symptom tokens, e.g. ['high_fever','headache']")
    severity: Optional[Dict[str, str]] = Field(
        None, description="Optional per-symptom severity: {'high_fever': 'severe'}. Values: mild/moderate/severe"
    )
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None, description="male / female / other")
    pregnant: Optional[bool] = None

    def validated_severity(self):
        if not self.severity:
            return None
        bad = {v for v in self.severity.values() if v not in VALID_SEVERITIES}
        if bad:
            raise HTTPException(status_code=400, detail=f"Invalid severity values: {bad}. Use mild/moderate/severe.")
        return self.severity


class PredictResponse(BaseModel):
    predicted_disease: str
    confidence: float
    low_confidence: bool
    recommended_keph_level: int
    base_keph_level: int
    context_adjustments: List[str]
    red_flags: List[dict]
    is_emergency: bool
    emergency_number: Optional[str] = None
    top_alternatives: List[dict]


class NearestFacilitiesRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    min_keph_level: Optional[int] = Field(None, ge=2, le=6)
    limit: int = Field(5, ge=1, le=50)


class TriageRequest(BaseModel):
    symptoms: List[str]
    severity: Optional[Dict[str, str]] = None
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = None
    pregnant: Optional[bool] = None
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    limit: int = Field(5, ge=1, le=50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def vectorize_symptoms(symptoms: List[str]) -> np.ndarray:
    vec = np.zeros((1, len(symptom_vocab)), dtype=np.uint8)
    unknown = []
    for s in symptoms:
        s = s.strip()
        if s in symptom_index:
            vec[0, symptom_index[s]] = 1
        else:
            unknown.append(s)
    return vec, unknown


def run_prediction(
    symptoms: List[str],
    severity: Optional[Dict[str, str]] = None,
    age: Optional[int] = None,
    sex: Optional[str] = None,
    pregnant: Optional[bool] = None,
    log_location: Optional[tuple] = None,
) -> PredictResponse:
    if sex is not None and sex not in VALID_SEX:
        raise HTTPException(status_code=400, detail=f"Invalid sex value. Use one of {VALID_SEX}.")

    vec, unknown = vectorize_symptoms(symptoms)
    if vec.sum() == 0:
        raise HTTPException(
            status_code=400,
            detail=f"None of the provided symptoms were recognized. Unknown: {unknown}",
        )

    # --- Red flags run independently of the ML model's confidence ---
    red_flags = check_red_flags(symptoms)
    is_emergency = len(red_flags) > 0

    # --- ML prediction ---
    proba = classifier.predict_proba(vec)[0]
    classes = classifier.classes_
    order = np.argsort(proba)[::-1]

    top_disease = classes[order[0]]
    top_confidence = float(proba[order[0]])
    low_confidence = top_confidence < CONFIDENCE_FLOOR
    base_keph_level = disease_keph_map.get(top_disease, 3)

    # --- Patient-context adjustment (age / pregnancy / severity) ---
    adjustment = adjust_keph_level(
        base_level=base_keph_level,
        symptoms=symptoms,
        age=age,
        sex=sex,
        pregnant=pregnant,
        severity=severity,
    )
    recommended_level = adjustment["adjusted_level"]

    # Emergency always routes to the highest capable tier regardless of the model's guess
    if is_emergency:
        recommended_level = max(recommended_level, 5)

    alternatives = [
        {"disease": classes[i], "confidence": round(float(proba[i]), 4)}
        for i in order[1:4]
        if proba[i] > 0
    ]

    prediction_log.log_prediction(
        predicted_disease=top_disease,
        confidence=top_confidence,
        recommended_keph_level=recommended_level,
        symptom_count=len(symptoms),
        red_flag_triggered=is_emergency,
        lat=log_location[0] if log_location else None,
        lon=log_location[1] if log_location else None,
    )

    return PredictResponse(
        predicted_disease=top_disease,
        confidence=round(top_confidence, 4),
        low_confidence=low_confidence,
        recommended_keph_level=recommended_level,
        base_keph_level=base_keph_level,
        context_adjustments=adjustment["reasons"],
        red_flags=red_flags,
        is_emergency=is_emergency,
        emergency_number=EMERGENCY_NUMBER_KE if is_emergency else None,
        top_alternatives=alternatives,
    )


def get_all_facilities():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM facilities").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "symptoms_known": len(symptom_vocab), "diseases_known": len(disease_keph_map)}


@app.get("/api/symptoms")
@limiter.limit("60/minute")
def list_symptoms(request: Request):
    return {"symptoms": symptom_vocab}


@app.get("/api/facilities")
@limiter.limit("20/minute")
def list_facilities(request: Request):
    return {"facilities": get_all_facilities()}


@app.get("/api/facility-stats")
@limiter.limit("20/minute")
def facility_stats(request: Request):
    """
    Real aggregate statistics over the facility registry, computed directly
    from the database (not hardcoded), for the frontend's Data Insights section.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM facilities").fetchone()[0]

    by_keph = cur.execute(
        "SELECT keph_level, COUNT(*) FROM facilities GROUP BY keph_level ORDER BY keph_level"
    ).fetchall()

    n_counties = cur.execute("SELECT COUNT(DISTINCT county) FROM facilities").fetchone()[0]

    top_counties = cur.execute(
        "SELECT county, COUNT(*) c FROM facilities GROUP BY county ORDER BY c DESC LIMIT 5"
    ).fetchall()

    conn.close()

    keph_labels = {
        2: "Dispensary", 3: "Health Centre", 4: "Primary/Sub-County Hospital",
        5: "County Referral Hospital", 6: "National Referral Hospital",
    }

    return {
        "total_facilities": total,
        "n_counties": n_counties,
        "by_keph_level": [
            {"keph_level": lvl, "label": keph_labels.get(lvl, f"Level {lvl}"), "count": count}
            for lvl, count in by_keph
        ],
        "top_counties": [{"county": c, "count": n} for c, n in top_counties],
    }


@app.get("/api/stats")
@limiter.limit("20/minute")
def usage_stats(request: Request):
    return prediction_log.get_usage_stats()


@app.get("/api/model-performance")
@limiter.limit("20/minute")
def model_performance(request: Request):
    """
    Returns the saved evaluation report from ML/evaluate_model.py
    (accuracy, cross-validation, per-class metrics, feature importance).
    Chart images are served separately from /model-assets/.
    """
    metrics_path = os.path.join(EVAL_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="No evaluation report found. Run `python ML/evaluate_model.py` first.",
        )
    with open(metrics_path) as f:
        return json.load(f)


@app.get("/api/model-comparison")
@limiter.limit("20/minute")
def model_comparison(request: Request):
    """
    Returns the multi-model comparison report from ML/evaluate_model_comparison.py
    (Logistic Regression vs Random Forest vs Gradient Boosting, same CV folds).
    """
    path = os.path.join(EVAL_DIR, "model_comparison.json")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="No model comparison found. Run `python ML/evaluate_model_comparison.py` first.",
        )
    with open(path) as f:
        return json.load(f)


@app.get("/api/robustness-evaluation")
@limiter.limit("20/minute")
def robustness_evaluation(request: Request):
    """
    Returns ML/evaluation/robustness_metrics.json -- the noise-injection
    stress test. Ran on the RAW Kaggle dataset, not the augmented
    production dataset.
    """
    path = os.path.join(EVAL_DIR, "robustness_metrics.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No robustness evaluation found.")
    with open(path) as f:
        return json.load(f)


@app.get("/api/deduplication-evaluation")
@limiter.limit("20/minute")
def deduplication_evaluation(request: Request):
    """
    Returns ML/evaluation/deduplicated_metrics.json -- the data-leakage
    diagnosis on the raw Kaggle dataset. Also ran on the raw dataset.
    """
    path = os.path.join(EVAL_DIR, "deduplicated_metrics.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No deduplication evaluation found.")
    with open(path) as f:
        return json.load(f)


@app.post("/api/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
def predict(request: Request, req: PredictRequest):
    severity = req.validated_severity()
    return run_prediction(
        symptoms=req.symptoms,
        severity=severity,
        age=req.age,
        sex=req.sex,
        pregnant=req.pregnant,
    )


@app.post("/api/nearest-facilities")
@limiter.limit("60/minute")
def nearest_facilities(request: Request, req: NearestFacilitiesRequest):
    facilities = get_all_facilities()
    ranked, routing_method = rank_by_road_distance(
        req.latitude, req.longitude, facilities,
        min_keph_level=req.min_keph_level, limit=req.limit
    )
    return {"facilities": ranked, "routing_method": routing_method}


@app.post("/api/triage")
@limiter.limit("30/minute")
def triage(request: Request, req: TriageRequest):
    if req.severity:
        bad = {v for v in req.severity.values() if v not in VALID_SEVERITIES}
        if bad:
            raise HTTPException(status_code=400, detail=f"Invalid severity values: {bad}")

    prediction = run_prediction(
        symptoms=req.symptoms,
        severity=req.severity,
        age=req.age,
        sex=req.sex,
        pregnant=req.pregnant,
        log_location=(req.latitude, req.longitude),
    )
    facilities = get_all_facilities()
    ranked, routing_method = rank_by_road_distance(
        req.latitude,
        req.longitude,
        facilities,
        min_keph_level=prediction.recommended_keph_level,
        limit=req.limit,
    )
    return {
        "prediction": prediction,
        "recommended_facilities": ranked,
        "routing_method": routing_method,
    }


# ---------------------------------------------------------------------------
# Serve frontend (so the whole app "opens into a link")
# ---------------------------------------------------------------------------
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

if os.path.isdir(EVAL_DIR):
    app.mount("/model-assets", StaticFiles(directory=EVAL_DIR), name="model-assets")
