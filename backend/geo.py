"""
geo.py
-------
Haversine distance utility for routing patients to the nearest appropriate
KEPH-level facility based on their GPS coordinates.
"""

import math
import re


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two (lat, lon) points in kilometers.
    """
    R = 6371.0088  # mean Earth radius, km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# Specialist or non-clinical listings that should not come up as a place to
# go for general symptoms (eye, dental, blood bank, pharmacies, physio).
_SPECIALIST_TYPES = {"dental clinic", "eye centre", "eye clinic", "regional blood transfusion centre"}
_SPECIALIST_NAME = re.compile(
    r"dental|dentist|optic|optician|\beye\b|pharmacy|chemist|physiotherap|blood (transfusion|satellite)",
    re.IGNORECASE,
)


def is_general_care(f):
    """False for dental, eye, blood-bank, pharmacy and physio listings."""
    if str(f.get("facility_type", "")).strip().lower() in _SPECIALIST_TYPES:
        return False
    return not _SPECIALIST_NAME.search(str(f.get("facility_name", "")))


def rank_facilities_by_distance(user_lat, user_lon, facilities, min_keph_level=None):
    """
    facilities: list of dicts with at least 'latitude', 'longitude', 'keph_level'
    min_keph_level: if given, only facilities with keph_level >= this are considered
                     (used to route a patient toward a facility capable of handling
                      their condition, not just the nearest dispensary)

    Returns facilities sorted by distance ascending, each with a 'distance_km' field.
    """
    candidates = [f for f in facilities if is_general_care(f)]
    if min_keph_level is not None:
        candidates = [f for f in candidates if f["keph_level"] >= min_keph_level]

    ranked = []
    for f in candidates:
        d = haversine_km(user_lat, user_lon, f["latitude"], f["longitude"])
        ranked.append({**f, "distance_km": round(d, 2)})

    ranked.sort(key=lambda f: f["distance_km"])
    return ranked
