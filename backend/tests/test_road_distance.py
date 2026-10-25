"""UT-02, UT-03, UT-04: ranking and road routing, with the routing service replaced by a mock."""
import road_distance
from geo import rank_facilities_by_distance

FACILITIES = [
    {"name": "A", "latitude": -1.2900, "longitude": 36.8200, "keph_level": 2},
    {"name": "B", "latitude": -1.3000, "longitude": 36.8300, "keph_level": 5},
    {"name": "C", "latitude": -1.2700, "longitude": 36.8000, "keph_level": 6},
    {"name": "D", "latitude": -1.4000, "longitude": 36.9000, "keph_level": 4},
]
USER = (-1.2864, 36.8172)


def test_ranking_is_nearest_first():
    ranked = rank_facilities_by_distance(*USER, FACILITIES)
    distances = [f["distance_km"] for f in ranked]
    assert distances == sorted(distances)


def test_ranking_with_minimum_level():
    ranked = rank_facilities_by_distance(*USER, FACILITIES, min_keph_level=5)
    assert ranked and all(f["keph_level"] >= 5 for f in ranked)


def test_road_ranking_uses_mock_and_sorts_by_road_distance(monkeypatch):
    def fake(lat, lon, candidates):
        # make the geographically nearest facility the longest by road
        return [(30.0 - 5 * i, 10.0 + i) for i, _ in enumerate(candidates)]

    monkeypatch.setattr(road_distance, "_fetch_road_distances", fake)
    ranked, method = road_distance.rank_by_road_distance(*USER, FACILITIES, limit=3)
    assert method == "road_network"
    assert len(ranked) == 3
    assert [f["distance_km"] for f in ranked] == sorted(f["distance_km"] for f in ranked)
    assert all("duration_min" in f and "straight_line_km" in f for f in ranked)


def test_routing_timeout_falls_back_to_straight_line(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("routing service timed out")

    monkeypatch.setattr(road_distance, "_fetch_road_distances", boom)
    ranked, method = road_distance.rank_by_road_distance(*USER, FACILITIES, limit=3)
    assert method == "haversine_fallback"
    assert len(ranked) == 3


def test_no_candidates_returns_empty_list(monkeypatch):
    ranked, method = road_distance.rank_by_road_distance(*USER, FACILITIES, min_keph_level=6 + 1)
    assert ranked == [] and method == "haversine_fallback"
