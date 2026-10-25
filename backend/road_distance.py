"""
road_distance.py
-------------------
Real road-network distance and travel-time routing, replacing pure
straight-line (Haversine) distance for the final ranking of facilities.

WHY THIS MATTERS: Haversine gives the distance "as the crow flies." Two
facilities 2km apart in a straight line could be a 3-minute drive or a
25-minute drive depending on rivers, highways, or lack of direct roads
between them. For a system recommending where a patient should actually
go, road distance/time is the more honest and more useful number.

TECHNIQUE: Uses OSRM (Open Source Routing Machine)'s public demo server,
which computes routes over real OpenStreetMap road network data --
free, no API key required. Specifically the "table" service, which
computes distances/durations from ONE origin to MANY destinations in a
single request (much faster than calling a routing API once per facility).

STRATEGY (two-stage, for both correctness and speed):
    1. Use Haversine (fast, no network call) to narrow ~9,700 facilities
       down to a small candidate pool (the nearest ~15-20 by straight-line
       distance).
    2. Call OSRM's table service ONCE with that candidate pool to get real
       road distances and drive times, then re-rank by road distance.

FALLBACK: If OSRM is unreachable (offline, demo server down/rate-limited,
timeout), this automatically falls back to the Haversine ranking that
already worked before -- the app never breaks, it just silently loses the
"real road distance" upgrade for that one request. The API response
always tells the frontend which method was actually used
(`routing_method`: "road_network" or "haversine_fallback"), so nothing is
presented as more precise than it actually is.

Usage:
    from road_distance import rank_by_road_distance

    ranked, method = rank_by_road_distance(user_lat, user_lon, candidate_facilities)
    # candidate_facilities: list of dicts with 'latitude', 'longitude', plus whatever
    # other fields the caller wants to keep (facility_name, keph_level, etc.)
    # ranked: same dicts, each with 'distance_km' and 'duration_min' added,
    #         sorted nearest-first by ROAD distance/time
    # method: "road_network" or "haversine_fallback"
"""

import requests

from geo import haversine_km, rank_facilities_by_distance

OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/{coords}"
OSRM_TIMEOUT_SECONDS = 4  # keep this short -- a slow external call should never make the app feel broken
MAX_CANDIDATES_FOR_ROAD_LOOKUP = 20  # how many Haversine-nearest facilities to send to OSRM


def _build_osrm_url(origin_lat, origin_lon, candidates):
    """OSRM expects lon,lat order (not lat,lon), semicolon-separated."""
    coords = [f"{origin_lon},{origin_lat}"]
    for c in candidates:
        coords.append(f"{c['longitude']},{c['latitude']}")
    coord_str = ";".join(coords)
    n = len(candidates)
    destinations = ";".join(str(i) for i in range(1, n + 1))
    return f"{OSRM_TABLE_URL.format(coords=coord_str)}?sources=0&destinations={destinations}&annotations=distance,duration"


def _fetch_road_distances(origin_lat, origin_lon, candidates):
    """
    Returns a list of (distance_km, duration_min) tuples, one per candidate,
    in the same order as `candidates`. Raises on any failure -- caller is
    responsible for catching and falling back to Haversine.
    """
    if not candidates:
        return []

    url = _build_osrm_url(origin_lat, origin_lon, candidates)
    resp = requests.get(url, timeout=OSRM_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM returned non-Ok status: {data.get('code')}")

    distances_m = data["distances"][0]   # single source row
    durations_s = data["durations"][0]

    n_candidates = len(candidates)
    # Defensive: OSRM's response should have exactly one entry per requested
    # destination (we explicitly pass destinations=1..N, excluding the source
    # itself at index 0). But if a future OSRM version or a different server
    # ever includes a leading self-distance entry, trim it rather than
    # silently misaligning every result by one position.
    if len(distances_m) == n_candidates + 1:
        distances_m = distances_m[1:]
        durations_s = durations_s[1:]

    if len(distances_m) != n_candidates:
        raise RuntimeError(
            f"OSRM response length ({len(distances_m)}) doesn't match candidate count ({n_candidates})"
        )

    results = []
    for dist_m, dur_s in zip(distances_m, durations_s):
        if dist_m is None or dur_s is None:
            # OSRM couldn't find a road route to this specific point
            # (e.g. an island, or a facility coordinate that's genuinely
            # off any mapped road) -- treat as unreachable, caller can
            # decide to drop it or fall back for this one candidate.
            results.append((None, None))
        else:
            results.append((round(dist_m / 1000, 2), round(dur_s / 60, 1)))
    return results


def rank_by_road_distance(user_lat, user_lon, facilities, min_keph_level=None, limit=6):
    """
    Two-stage ranking: Haversine pre-filter -> OSRM road-distance re-rank.
    Falls back cleanly to pure Haversine ranking if OSRM is unavailable.

    Returns (ranked_list, method) where method is "road_network" or "haversine_fallback".
    """
    # Stage 1: cheap Haversine pre-filter down to a small candidate pool
    haversine_ranked = rank_facilities_by_distance(
        user_lat, user_lon, facilities, min_keph_level=min_keph_level
    )
    candidates = haversine_ranked[:MAX_CANDIDATES_FOR_ROAD_LOOKUP]

    if not candidates:
        return [], "haversine_fallback"

    # Stage 2: try real road distances for that candidate pool
    try:
        road_results = _fetch_road_distances(user_lat, user_lon, candidates)

        enriched = []
        for facility, (dist_km, dur_min) in zip(candidates, road_results):
            if dist_km is None:
                continue  # unreachable by road per OSRM -- skip rather than show a wrong number
            enriched.append({
                **facility,
                "distance_km": dist_km,
                "duration_min": dur_min,
                "straight_line_km": facility["distance_km"],  # keep the Haversine value too, for comparison
            })

        if not enriched:
            # every candidate came back unreachable (unlikely, but be safe)
            return candidates[:limit], "haversine_fallback"

        enriched.sort(key=lambda f: f["distance_km"])
        return enriched[:limit], "road_network"

    except Exception:
        # Network error, timeout, OSRM down, rate-limited, malformed response --
        # whatever it is, don't let it break the request. Fall back silently.
        return candidates[:limit], "haversine_fallback"
