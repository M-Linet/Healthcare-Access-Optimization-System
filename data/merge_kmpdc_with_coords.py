"""
merge_kmpdc_with_coords.py (v2 -- fixed column names)
---------------------------------------------------------
FIXED: the previous version assumed data/healthcare_facilities.csv already
had clean column names like 'facility_name' and 'keph_level'. It doesn't --
those clean names are only created later, inside backend/init_db.py, when
it builds the SQLite database. The raw CSV (the actual energydata.info
export) uses the ORIGINAL government column headers instead:
    OBJECTID, Facility_N, Type, Owner, County, Sub_County, Division,
    Location, Sub_Locati, Constituen, Nearest_To, Latitude, Longitude

This version reads those correctly, and does NOT need a 'keph_level'
column to already exist -- there isn't one in the raw file (KEPH level is
only inferred later, inside init_db.py, from the 'Type' column).

STRATEGY: rather than duplicating init_db.py's inference logic, this
script takes a different approach that requires ZERO changes to
init_db.py: for every facility where a confident KMPDC match is found, it
overwrites the 'Type' column with a representative type string that
init_db.py's own TYPE_TO_KEPH dictionary already maps to the correct KEPH
level (e.g., a KMPDC "LEVEL 6A" match sets Type to "National Referral
Hospital", which init_db.py already maps to KEPH level 6). The output CSV
keeps the exact same column structure as the original -- you can drop it
straight in as a replacement and run init_db.py exactly as before, no
other changes needed anywhere.

Usage:
    pip install pandas rapidfuzz
    python data\\scrape_kmpdc.py                    (produces kmpdc_all_facilities.csv)
    python data\\merge_kmpdc_with_coords.py          (this script)
    (run both from the project ROOT)

Output:
    data/healthcare_facilities_enriched.csv
"""

import re
import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:
    raise SystemExit("This script needs rapidfuzz. Install it with: pip install rapidfuzz")

COORDS_PATH = "data/healthcare_facilities.csv"
KMPDC_PATH = "data/kmpdc_all_facilities.csv"
OUT_PATH = "data/healthcare_facilities_enriched.csv"

FUZZY_THRESHOLD = 90

# Must exactly match backend/init_db.py's TYPE_TO_KEPH keys, so the
# enriched Type value maps to the intended KEPH level when init_db.py
# processes this file. One representative Type string per level.
KEPH_TO_TYPE = {
    2: "Dispensary",
    3: "Health Centre",
    4: "District Hospital",
    5: "Provincial General Hospital",
    6: "National Referral Hospital",
}


def read_csv_robust(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        print(f"  Note: {path} is not UTF-8 encoded, retrying as ISO-8859-1 (Latin-1)...")
        return pd.read_csv(path, encoding="ISO-8859-1")


def normalize_name(name):
    if pd.isna(name):
        return ""
    n = str(name).upper()
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\b(SUB[\s-]?COUNTY|COUNTY|REFERRAL|HOSPITAL|CENTRE|CENTER|DISPENSARY|CLINIC)\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def main():
    print("Loading coordinate-bearing facilities (energydata.info)...")
    coords_df = read_csv_robust(COORDS_PATH)
    print(f"  {len(coords_df)} facilities loaded")
    print(f"  Columns found: {list(coords_df.columns)}")

    name_col = "Facility_N" if "Facility_N" in coords_df.columns else "facility_name"
    county_col = "County" if "County" in coords_df.columns else "county"

    if name_col not in coords_df.columns:
        raise SystemExit(
            f"Could not find a facility name column. Expected 'Facility_N' or 'facility_name'. "
            f"Actual columns: {list(coords_df.columns)}"
        )

    print("Loading KMPDC official facility levels...")
    kmpdc_df = read_csv_robust(KMPDC_PATH)
    kmpdc_df = kmpdc_df.dropna(subset=["keph_level"])
    print(f"  {len(kmpdc_df)} KMPDC facilities with a parseable official level")

    coords_df["_norm_name"] = coords_df[name_col].apply(normalize_name)
    kmpdc_df["_norm_name"] = kmpdc_df["Facility Name"].apply(normalize_name)

    kmpdc_by_county = {
        county: group for county, group in kmpdc_df.groupby(kmpdc_df["County"].astype(str).str.upper())
    }

    keph_source = []
    match_scores = []
    matched_count = 0
    original_type_col = coords_df["Type"].copy() if "Type" in coords_df.columns else None

    for idx, row in coords_df.iterrows():
        county = str(row.get(county_col, "")).upper()
        candidates = kmpdc_by_county.get(county)

        best_score = 0
        best_level = None

        if candidates is not None and len(candidates) > 0:
            exact = candidates[candidates["_norm_name"] == row["_norm_name"]]
            if len(exact) > 0 and row["_norm_name"]:
                best_score = 100
                best_level = int(exact.iloc[0]["keph_level"])
            else:
                for _, cand in candidates.iterrows():
                    if not row["_norm_name"] or not cand["_norm_name"]:
                        continue
                    score = fuzz.token_sort_ratio(row["_norm_name"], cand["_norm_name"])
                    if score > best_score:
                        best_score = score
                        best_level = int(cand["keph_level"])

        if best_score >= FUZZY_THRESHOLD and best_level is not None and "Type" in coords_df.columns:
            coords_df.at[idx, "Type"] = KEPH_TO_TYPE.get(best_level, coords_df.at[idx, "Type"])
            keph_source.append("KMPDC_official")
            match_scores.append(best_score)
            matched_count += 1
        else:
            keph_source.append("inferred")
            match_scores.append(best_score)

    coords_df["keph_source"] = keph_source
    coords_df["match_score"] = match_scores
    coords_df = coords_df.drop(columns=["_norm_name"])

    coords_df.to_csv(OUT_PATH, index=False)

    print(f"\nMatched {matched_count} of {len(coords_df)} facilities to an official KMPDC level "
          f"({matched_count/len(coords_df)*100:.1f}%)")
    print(f"Saved -> {OUT_PATH}")
    print("\nThis file has the SAME column structure as the original healthcare_facilities.csv --")
    print("to use it: backup the original, then copy this file over it, then re-run:")
    print("  python backend\\init_db.py")


if __name__ == "__main__":
    main()