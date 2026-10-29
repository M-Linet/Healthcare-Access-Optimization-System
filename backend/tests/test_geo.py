"""
test_geo.py
------------
Unit tests for backend/geo.py — Haversine distance and facility ranking.
Run with: pytest tests/test_geo.py -v
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geo import haversine_km, rank_facilities_by_distance


def test_haversine_zero_distance():
    """Same point should be 0km apart."""
    d = haversine_km(-1.2864, 36.8172, -1.2864, 36.8172)
    assert d == 0.0


def test_haversine_known_distance():
    """Nairobi CBD to JKIA airport is roughly 15-16km."""
    nairobi_cbd = (-1.2864, 36.8172)
    jkia = (-1.3192, 36.9278)
    d = haversine_km(*nairobi_cbd, *jkia)
    assert 11 <= d <= 16, f"Expected ~11-16km straight-line, got {d}"


def test_haversine_symmetry():
    """Distance A->B should equal distance B->A."""
    a = (-1.2864, 36.8172)
    b = (-0.0917, 34.7680)
    d1 = haversine_km(*a, *b)
    d2 = haversine_km(*b, *a)
    assert math.isclose(d1, d2, rel_tol=1e-9)


def test_haversine_antipodal_sanity():
    """Distance should never exceed half of Earth's circumference (~20037km)."""
    d = haversine_km(0, 0, 0, 180)
    assert d <= 20038


def _sample_facilities():
    return [
        {"facility_id": "A", "facility_name": "Near Dispensary", "keph_level": 2, "latitude": -1.28, "longitude": 36.81},
        {"facility_id": "B", "facility_name": "Mid Health Centre", "keph_level": 3, "latitude": -1.30, "longitude": 36.85},
        {"facility_id": "C", "facility_name": "Far National Hospital", "keph_level": 6, "latitude": 0.05, "longitude": 37.65},
    ]


def test_rank_facilities_sorted_by_distance():
    facilities = _sample_facilities()
    ranked = rank_facilities_by_distance(-1.2864, 36.8172, facilities)
    distances = [f["distance_km"] for f in ranked]
    assert distances == sorted(distances), "Facilities should be sorted nearest-first"


def test_rank_facilities_min_keph_filter():
    facilities = _sample_facilities()
    ranked = rank_facilities_by_distance(-1.2864, 36.8172, facilities, min_keph_level=5)
    assert len(ranked) == 1
    assert ranked[0]["facility_id"] == "C"


def test_rank_facilities_no_filter_returns_all():
    facilities = _sample_facilities()
    ranked = rank_facilities_by_distance(-1.2864, 36.8172, facilities)
    assert len(ranked) == 3


def test_rank_facilities_empty_list():
    ranked = rank_facilities_by_distance(-1.2864, 36.8172, [])
    assert ranked == []


def test_specialist_clinics_excluded():
    from geo import is_general_care
    assert not is_general_care({"facility_type": "Dental Clinic", "facility_name": "X"})
    assert not is_general_care({"facility_type": "Medical Clinic", "facility_name": "Tasneem Pharmacy"})
    assert is_general_care({"facility_type": "Health Centre", "facility_name": "Kibera Health Centre"})
    fac = [
        {"latitude": -1.28, "longitude": 36.82, "keph_level": 3, "facility_type": "Eye Centre", "facility_name": "Bans Optical"},
        {"latitude": -1.30, "longitude": 36.82, "keph_level": 3, "facility_type": "Health Centre", "facility_name": "Real HC"},
    ]
    ranked = rank_facilities_by_distance(-1.28, 36.82, fac, min_keph_level=3)
    assert [f["facility_name"] for f in ranked] == ["Real HC"]


def test_only_real_national_referral_hospitals_are_level_6():
    import sqlite3, os
    db = os.path.join(os.path.dirname(__file__), "..", "database.sqlite")
    rows = sqlite3.connect(db).execute(
        "SELECT facility_name FROM facilities WHERE keph_level = 6").fetchall()
    names = " ".join(r[0].lower() for r in rows)
    for private in ("avenue hospital", "nairobi west", "eldoret hospital"):
        assert private not in names
