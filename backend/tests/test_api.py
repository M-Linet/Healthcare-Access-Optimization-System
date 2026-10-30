"""Integration and security tests against the FastAPI app (IT-02, IT-03, IT-05, SC-01..SC-04, SC-02)."""
import pytest
from fastapi.testclient import TestClient

import main
import road_distance


@pytest.fixture(autouse=True)
def no_real_routing(monkeypatch):
    """Use the straight-line fallback so the tests do not need the internet."""
    def boom(*a, **k):
        raise TimeoutError("routing disabled in tests")
    monkeypatch.setattr(road_distance, "_fetch_road_distances", boom)
    main.limiter.reset()
    yield
    main.limiter.reset()


client = TestClient(main.app, raise_server_exceptions=False)
NAIROBI = {"latitude": -1.2864, "longitude": 36.8172}


def test_health():
    assert client.get("/health").status_code == 200


def test_predict_normal_case():  # IT-02
    r = client.post("/api/predict", json={"symptoms": ["high_fever", "headache", "vomiting"]})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["confidence"] <= 1
    assert 2 <= body["recommended_keph_level"] <= 6
    assert len(body["top_alternatives"]) <= 3


def test_predict_emergency():  # IT-03
    r = client.post("/api/predict", json={"symptoms": ["chest_pain", "breathlessness", "sweating"]})
    body = r.json()
    assert body["is_emergency"] is True
    assert body["emergency_number"] == "999 / 112"
    assert body["recommended_keph_level"] >= 5


def test_pregnancy_raises_level():  # VT-03
    r = client.post("/api/predict", json={"symptoms": ["abdominal_pain"], "pregnant": True, "sex": "female"})
    assert r.json()["recommended_keph_level"] >= 3


def test_nearest_facilities_fallback_still_returns_results():  # IT-04 (routing blocked)
    r = client.post("/api/nearest-facilities", json={**NAIROBI, "limit": 5})
    body = r.json()
    assert r.status_code == 200
    assert body["routing_method"] == "haversine_fallback"
    d = [f["distance_km"] for f in body["facilities"]]
    assert d == sorted(d) and len(d) == 5


def test_triage_facilities_meet_the_predicted_level():  # IT-05
    r = client.post("/api/triage", json={"symptoms": ["chest_pain", "breathlessness", "sweating"], **NAIROBI})
    body = r.json()
    level = body["prediction"]["recommended_keph_level"]
    assert body["recommended_facilities"]
    assert all(f["keph_level"] >= level for f in body["recommended_facilities"])


def test_security_headers_on_normal_and_error_responses():  # SC-01
    wanted = ["x-content-type-options", "x-frame-options", "referrer-policy",
              "permissions-policy", "content-security-policy"]
    ok = client.get("/health")
    err = client.post("/api/predict", json={"symptoms": ["not_a_symptom"]})
    for resp in (ok, err):
        for h in wanted:
            assert h in resp.headers, f"missing {h}"


@pytest.mark.parametrize("payload", [
    {"symptoms": ["high_fever"], "age": 200},
    {"symptoms": "high_fever"},
    {"symptoms": ["high_fever"], "severity": {"high_fever": "extreme"}},
    {"symptoms": ["high_fever"], "sex": "robot"},
])
def test_bad_predict_input_is_rejected_not_500(payload):  # SC-03
    r = client.post("/api/predict", json=payload)
    assert 400 <= r.status_code < 500


def test_bad_coordinates_rejected():  # SC-03
    r = client.post("/api/nearest-facilities", json={"latitude": 100, "longitude": 36.8})
    assert r.status_code == 422


def test_oversized_body_rejected_with_413():  # SC-03
    big = {"symptoms": ["high_fever"] * 30000}
    r = client.post("/api/predict", json=big)
    assert r.status_code == 413


def test_html_input_is_not_echoed_as_html():  # SC-04
    r = client.post("/api/predict", json={"symptoms": ["<script>alert(1)</script>"]})
    assert r.status_code == 400
    assert "<script>" not in r.text or r.headers["content-type"].startswith("application/json")


def test_rate_limit_returns_429_on_31st_request():  # SC-02
    codes = [client.post("/api/predict", json={"symptoms": ["high_fever"]}).status_code for _ in range(31)]
    assert codes[:30] == [200] * 30
    assert codes[30] == 429


def test_far_warning_for_remote_search():
    # Lodwar is far from any level 6 hospital
    r = client.post("/api/nearest-facilities",
                    json={"latitude": 3.1191, "longitude": 35.5973, "min_keph_level": 6, "limit": 3})
    body = r.json()
    assert body["far_warning"] is True
    assert body["nearest_km"] > 50


def test_no_far_warning_in_nairobi():
    r = client.post("/api/nearest-facilities", json={**NAIROBI, "min_keph_level": 6, "limit": 3})
    body = r.json()
    assert body["far_warning"] is False
    assert body["nearest_km"] < 50
