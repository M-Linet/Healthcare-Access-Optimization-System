"""
prediction_log.py
-------------------
Lightweight, non-PII logging of predictions to a local SQLite table, so the
system can show usage patterns / accuracy trends (e.g. in a project defense)
without storing anything that identifies a specific person.

Explicitly NOT logged: exact GPS coordinates, IP address, name, phone
number, or any free-text patient input. Coordinates are rounded to ~11km
precision (2 decimal places) before storage — enough for county-level
usage patterns, not enough to pinpoint an individual's home or exact location.
"""

import os
import sqlite3
from datetime import datetime, timezone

LOG_DB_PATH = os.path.join(os.path.dirname(__file__), "prediction_log.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    predicted_disease TEXT NOT NULL,
    confidence REAL NOT NULL,
    recommended_keph_level INTEGER NOT NULL,
    red_flag_triggered INTEGER NOT NULL DEFAULT 0,
    symptom_count INTEGER NOT NULL,
    approx_lat REAL,
    approx_lon REAL
);
"""


def _get_conn():
    conn = sqlite3.connect(LOG_DB_PATH)
    conn.execute(SCHEMA)
    return conn


def log_prediction(
    predicted_disease: str,
    confidence: float,
    recommended_keph_level: int,
    symptom_count: int,
    red_flag_triggered: bool = False,
    lat: float = None,
    lon: float = None,
):
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO prediction_log
            (timestamp, predicted_disease, confidence, recommended_keph_level,
             red_flag_triggered, symptom_count, approx_lat, approx_lon)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            predicted_disease,
            confidence,
            recommended_keph_level,
            int(red_flag_triggered),
            symptom_count,
            round(lat, 2) if lat is not None else None,
            round(lon, 2) if lon is not None else None,
        ),
    )
    conn.commit()
    conn.close()


def get_usage_stats():
    """Simple aggregate stats for a /api/stats endpoint or defense demo."""
    conn = _get_conn()
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]
    by_disease = cur.execute(
        "SELECT predicted_disease, COUNT(*) c FROM prediction_log GROUP BY predicted_disease ORDER BY c DESC LIMIT 10"
    ).fetchall()
    by_keph = cur.execute(
        "SELECT recommended_keph_level, COUNT(*) c FROM prediction_log GROUP BY recommended_keph_level ORDER BY recommended_keph_level"
    ).fetchall()
    red_flags = cur.execute(
        "SELECT COUNT(*) FROM prediction_log WHERE red_flag_triggered = 1"
    ).fetchone()[0]
    avg_confidence = cur.execute("SELECT AVG(confidence) FROM prediction_log").fetchone()[0]

    conn.close()
    return {
        "total_predictions": total,
        "top_predicted_diseases": [{"disease": d, "count": c} for d, c in by_disease],
        "by_keph_level": [{"keph_level": k, "count": c} for k, c in by_keph],
        "red_flag_triggers": red_flags,
        "average_confidence": round(avg_confidence, 4) if avg_confidence else None,
    }
