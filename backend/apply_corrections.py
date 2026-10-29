"""
apply_corrections.py
Applies the documented fixes in data/facility_corrections.csv to
backend/database.sqlite. Each row says: facility_id, field, old_value,
new_value, reason. A change is only made if the current value equals
old_value, so the script is safe to run twice.

Run after init_db.py (which rebuilds the table from the raw CSV):
    python backend/apply_corrections.py
"""
import csv
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE / "database.sqlite"
CSV = HERE.parent / "data" / "facility_corrections.csv"
ALLOWED = {"keph_level", "latitude", "longitude", "facility_type", "ownership"}


def main():
    conn = sqlite3.connect(DB)
    applied = skipped = 0
    with open(CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            field = row["field"]
            if field not in ALLOWED:
                raise SystemExit(f"Field not allowed: {field}")
            cur = conn.execute(
                f"SELECT {field} FROM facilities WHERE facility_id = ?", (row["facility_id"],)
            ).fetchone()
            if cur is None:
                print(f"missing   {row['facility_id']}")
                skipped += 1
            elif str(cur[0]) == row["old_value"]:
                conn.execute(
                    f"UPDATE facilities SET {field} = ? WHERE facility_id = ?",
                    (row["new_value"], row["facility_id"]),
                )
                print(f"applied   {row['facility_id']} {field}: {row['old_value']} -> {row['new_value']}")
                applied += 1
            else:
                print(f"skipped   {row['facility_id']} {field}: now {cur[0]}, expected {row['old_value']}")
                skipped += 1
    conn.commit()
    conn.close()
    print(f"Done: {applied} applied, {skipped} skipped.")


if __name__ == "__main__":
    main()
