"""
init_db.py
-----------
Initializes backend/database.sqlite from the real facility registry:
data/healthcare_facilities.csv (World Bank / energydata.info "Kenya -
Healthcare Facilities" export, sourced from opendata.go.ke, 10,013 facilities
across all 47 counties).

That source file doesn't carry a KEPH level directly — it lists a raw
`Type` field (Dispensary, Health Centre, District Hospital, ...). We map
`Type` -> KEPH level 2-6 using Kenya's Essential Package for Health
definitions (see TYPE_TO_KEPH below). Unrecognized/ancillary types (e.g.
stand-alone laboratories, VCT centres, funeral homes) are excluded from the
routing table since they aren't a place to route a walk-in patient's primary
care visit.

Run this whenever the facility CSV changes to rebuild the facilities table.

Usage:
    python init_db.py
"""

import csv
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "database.sqlite")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "healthcare_facilities.csv")
CSV_ENCODING = "ISO-8859-1"  # source file is not UTF-8

SCHEMA = """
CREATE TABLE IF NOT EXISTS facilities (
    facility_id     TEXT PRIMARY KEY,
    facility_name   TEXT NOT NULL,
    keph_level      INTEGER NOT NULL,
    facility_type   TEXT,
    county          TEXT,
    sub_county      TEXT,
    latitude        REAL NOT NULL,
    longitude       REAL NOT NULL,
    services_offered TEXT,
    ownership       TEXT
);
"""

# Kenya Essential Package for Health (KEPH) level mapping.
# Levels: 2=Dispensary, 3=Health Centre, 4=Primary/Sub-County Hospital,
#         5=County Referral Hospital, 6=National Referral Hospital
TYPE_TO_KEPH = {
    "Dispensary": 2,
    "Medical Clinic": 2,
    "Dental Clinic": 2,
    "Eye Clinic": 2,
    "Maternity Home": 2,
    "Health Centre": 3,
    "Medical Centre": 3,
    "Nursing Home": 3,
    "Eye Centre": 3,
    "Sub-District Hospital": 4,
    "District Hospital": 4,
    "Other Hospital": 4,
    "Rural Health Training Centre": 4,
    "Provincial General Hospital": 5,
    "Regional Blood Transfusion Centre": 5,
    "National Referral Hospital": 6,
}

# Types with no meaningful KEPH triage role (ancillary services, admin
# offices, standalone diagnostics) are skipped when building the routing
# table — they aren't places to send a walk-in patient.
EXCLUDED_TYPES = {
    "VCT Centre (Stand-Alone)", "Laboratory (Stand-alone)", "Not in List",
    "Health Programme", "Radiology Unit", "Health Project",
    "Training Institution in Health (Stand-alone)", "District Health Office",
    "Blood Bank", "Funeral Home (Stand-alone)",
}


def load_source_rows(csv_path):
    with open(csv_path, newline="", encoding=CSV_ENCODING) as f:
        reader = csv.DictReader(f)
        return list(reader)


def transform_row(row, idx):
    ftype = row["Type"].strip()
    if ftype in EXCLUDED_TYPES or ftype not in TYPE_TO_KEPH:
        return None
    try:
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
    except (ValueError, KeyError):
        return None
    if lat == 0.0 and lon == 0.0:
        return None

    return {
        "facility_id": f"EDI{row.get('OBJECTID', idx).strip() if isinstance(row.get('OBJECTID'), str) else idx}",
        "facility_name": row["Facility_N"].strip(),
        "keph_level": TYPE_TO_KEPH[ftype],
        "facility_type": ftype,
        "county": row["County"].strip(),
        "sub_county": row["Sub_County"].strip(),
        "latitude": lat,
        "longitude": lon,
        "services_offered": ftype,
        "ownership": row["Owner"].strip(),
    }


def init_db(csv_path=CSV_PATH, db_path=DB_PATH):
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(SCHEMA)

    source_rows = load_source_rows(csv_path)
    transformed = []
    skipped = 0
    for i, row in enumerate(source_rows):
        t = transform_row(row, i)
        if t is None:
            skipped += 1
        else:
            transformed.append(t)

    cur.executemany(
        """
        INSERT OR IGNORE INTO facilities (
            facility_id, facility_name, keph_level, facility_type,
            county, sub_county, latitude, longitude, services_offered, ownership
        ) VALUES (:facility_id, :facility_name, :keph_level, :facility_type,
                  :county, :sub_county, :latitude, :longitude, :services_offered, :ownership)
        """,
        transformed,
    )
    conn.commit()

    count = cur.execute("SELECT COUNT(*) FROM facilities").fetchone()[0]
    print(f"Initialized {db_path} with {count} facilities ({skipped} rows skipped: ancillary type or missing coords).")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_keph_level ON facilities(keph_level)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_county ON facilities(county)")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
