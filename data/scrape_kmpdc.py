"""
scrape_kmpdc.py (v2 -- more defensive, clearer status reporting)
--------------------------------------------------------------------
Pulls the OFFICIAL Kenya Medical Practitioners and Dentists Council (KMPDC)
health facility registers -- public and private -- and saves them as clean
CSVs with the regulator's own facility level (not inferred).

This version wraps everything in error handling and writes a short,
one-line status to SCRAPE_STATUS.txt at the end, no matter what happens --
so even if something fails, there's one tiny file to check instead of
needing to read a full terminal scrollback.

Usage:
    pip install requests pandas lxml html5lib beautifulsoup4
    python scrape_kmpdc.py

Output:
    data/kmpdc_public_facilities.csv
    data/kmpdc_private_facilities.csv
    data/kmpdc_all_facilities.csv
    SCRAPE_STATUS.txt   <-- check this one file for a one-line result
"""

import os
import io
import re
import sys
import time
import traceback

STATUS_FILE = "SCRAPE_STATUS.txt"


def write_status(message):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)


def main():
    try:
        import requests
    except ImportError:
        write_status("FAILED: 'requests' package not installed. Run: pip install requests")
        return

    try:
        import pandas as pd
    except ImportError:
        write_status("FAILED: 'pandas' package not installed. Run: pip install pandas")
        return

    URLS = {
        "public": "https://registers.kmpdc.go.ke/healthFacilities/getLicensedPublicHealthFacilities/",
        "private": "https://registers.kmpdc.go.ke/healthFacilities/getLicensedPrivateHealthFacilities/",
    }
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }

    os.makedirs("data", exist_ok=True)

    def clean_facility_level(raw_level):
        if pd.isna(raw_level):
            return None, None
        text = str(raw_level).upper()
        match = re.search(r"LEVEL\s*(\d)([A-Z]?)", text)
        if not match:
            return None, None
        return int(match.group(1)), (match.group(2) or "")

    def try_read_html(html_text):
        """Try multiple parser backends since not all may be installed.
        CRITICAL: wrap in io.StringIO() -- passing the raw string directly
        can make pandas/lxml misdetect it as a filename/path instead of
        literal HTML content, causing a crash whose error message embeds
        the entire HTML text (very confusing to diagnose from a terminal).
        """
        last_error = None
        for flavor in ("lxml", "html5lib", "bs4"):
            try:
                if flavor == "bs4":
                    return pd.read_html(io.StringIO(html_text))
                return pd.read_html(io.StringIO(html_text), flavor=flavor)
            except Exception as e:
                last_error = e
                continue
        raise last_error if last_error else RuntimeError("No HTML parser available")

    all_frames = []
    saved_files = []

    for label, url in URLS.items():
        try:
            print(f"Fetching {label} facilities from {url} ...")
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            print(f"  Got {len(resp.text):,} characters (status {resp.status_code})")

            tables = try_read_html(resp.text)
            if not tables:
                write_status(f"FAILED: No tables found on {label} page.")
                return

            df = max(tables, key=len)
            print(f"  Found {len(df)} rows in the main data table for {label}.")

            df.columns = [str(c).strip() for c in df.columns]
            if "Facility Name" in df.columns:
                df = df[df["Facility Name"] != "Facility Name"].copy()
            if "Status" in df.columns:
                before = len(df)
                df = df[df["Status"].astype(str).str.upper() == "ACTIVE"].copy()
                print(f"  Filtered to ACTIVE only: {len(df)} of {before} rows")
            if "Facility Level" in df.columns:
                levels = df["Facility Level"].apply(clean_facility_level)
                df["keph_level"] = [l[0] for l in levels]
                df["keph_subtier"] = [l[1] for l in levels]

            df["source"] = f"KMPDC_{label}"

            out_path = f"data/kmpdc_{label}_facilities.csv"
            df.to_csv(out_path, index=False)
            saved_files.append(out_path)
            print(f"  Saved -> {out_path} ({len(df)} rows)\n")

            all_frames.append(df)
            time.sleep(1)

        except Exception as e:
            tb = traceback.format_exc()
            with open("SCRAPE_ERROR_DETAIL.txt", "w", encoding="utf-8") as f:
                f.write(tb)
            write_status(f"FAILED on '{label}': {type(e).__name__}: {str(e)[:150]} "
                         f"(full traceback in SCRAPE_ERROR_DETAIL.txt)")
            return

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv("data/kmpdc_all_facilities.csv", index=False)
        saved_files.append("data/kmpdc_all_facilities.csv")
        write_status(f"SUCCESS: {len(combined)} total active facilities saved across {len(saved_files)} files.")
    else:
        write_status("FAILED: No data was fetched from either register.")


if __name__ == "__main__":
    main()