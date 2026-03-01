"""
wb_data_fetcher.py
==================
Fetches project evaluation data from World Bank IEG.
Falls back to local CSV if network unavailable.

Sources:
    Primary:  ieg.worldbankgroup.org — IEG project performance ratings CSV
    Fallback: datacatalogfiles.worldbank.org — World Bank data catalog mirror
    Offline:  data/ieg_ratings.csv — cached from previous live fetch

IMPORTANT: This module only reads. No local data is modified.
           Fetch timestamp is logged for audit trail.
"""

import os, time, logging
import requests, pandas as pd, numpy as np
from io import StringIO

log = logging.getLogger(__name__)

WB_IEG_URLS = [
    "https://ieg.worldbankgroup.org/sites/default/files/Data/reports/ieg_world_bank_project_performance_ratings.csv",
    "https://datacatalogfiles.worldbank.org/ddh-published/0037394/DR0045406/ieg_world_bank_project_performance_ratings.csv",
]
WB_PROJECTS_API = "https://search.worldbank.org/api/v2/projects"
LOCAL_DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "ieg_ratings.csv")


def fetch_ieg_data(max_projects: int = None, force_local: bool = False) -> pd.DataFrame:
    """
    Fetch WB IEG project ratings.

    Priority:
        1. Live IEG CSV from worldbankgroup.org
        2. WB Projects API (metadata only, no IEG ratings)
        3. Local cache at data/ieg_ratings.csv

    Returns DataFrame with standardised column names for lsr_classifier.
    """
    if not force_local:
        for url in WB_IEG_URLS:
            try:
                log.info(f"  Fetching: {url}")
                resp = requests.get(url, timeout=90)
                resp.raise_for_status()
                df = pd.read_csv(StringIO(resp.text))
                log.info(f"  ✓ {len(df):,} rows from IEG CSV")
                df = _standardise_columns(df)
                if max_projects:
                    df = df.head(max_projects)
                _cache(df)
                return df
            except Exception as e:
                log.warning(f"  IEG URL failed ({url[:50]}): {e}")

        try:
            log.info("  Trying Projects API...")
            df = _fetch_projects_api(max_projects)
            log.info(f"  ✓ {len(df):,} rows from Projects API")
            return _standardise_columns(df)
        except Exception as e:
            log.warning(f"  Projects API failed: {e}")

    if os.path.exists(LOCAL_DATA_PATH):
        log.info(f"  Using local cache: {LOCAL_DATA_PATH}")
        df = pd.read_csv(LOCAL_DATA_PATH)
        if max_projects:
            df = df.head(max_projects)
        log.info(f"  ✓ {len(df):,} rows from cache")
        return df

    raise RuntimeError(
        "No data available.\n"
        "  Option A: Connect to internet and rerun\n"
        "  Option B: Download IEG CSV manually:\n"
        "            https://ieg.worldbankgroup.org/evaluations/data\n"
        "            Save as: data/ieg_ratings.csv\n"
        "            Then: python run_pipeline.py --local"
    )


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map actual WB IEG CSV column names to internal LSR pipeline names.

    The IEG CSV uses abbreviated headers (QaE, QoS, Borr Perf, etc.)
    These MUST be mapped or lsr_classifier receives all-NaN rating columns
    and produces zero classified projects.
    """
    ieg_map = {
        # ── Core identifiers ──────────────────────────────────────────────────
        "Proj ID":              "project_id",
        "Project ID":           "project_id",
        "Proj Name":            "project_name",
        "Project Name":         "project_name",
        "Country":              "countryname",
        "Country Code":         "country_code",
        "Region":               "regionname",

        # ── Time ─────────────────────────────────────────────────────────────
        "Approval FY":          "boardapprovaldate",
        "Board Approval Date":  "boardapprovaldate",
        "Exit FY":              "closingdate",
        "Closing Date":         "closingdate",
        "IEG Exit Date":        "ieg_exit_date",

        # ── Finance ───────────────────────────────────────────────────────────
        "Commit Amt (USD)":     "totalcommamt",
        "Project Cost (USD)":   "totalcommamt",
        "Proj Cost":            "totalcommamt",
        "Lending Instr":        "lending_instrument",
        "Lending Instr Type":   "lending_instr_type",
        "Prod Line":            "prod_line",

        # ── Sector / theme ────────────────────────────────────────────────────
        "Sector (Code)":        "sector1",
        "Sector Board":         "sector1",
        "Sector":               "sector1",
        "Theme (Code)":         "theme",

        # ── IEG RATINGS — the four LSR components ─────────────────────────────
        # SHORT names (actual IEG CSV format):
        "QaE":                  "quality_at_entry",       # ← CRITICAL
        "QoS":                  "quality_of_supervision", # ← CRITICAL
        "Borr Perf":            "borrower_performance",   # ← CRITICAL
        "M&E Qual":             "me_quality",             # ← CRITICAL
        "Bank Perf":            "bank_performance",
        "Risk to Dev Outcome":  "risk_to_dev_outcome",
        "ICR Quality":          "icr_quality",

        # LONG names (alternate format, some downloads):
        "Quality at Entry":          "quality_at_entry",
        "Quality of Supervision":    "quality_of_supervision",
        "Borrower Performance":      "borrower_performance",
        "M&E Quality":               "me_quality",
        "Bank Performance":          "bank_performance",

        # ── Outcome ───────────────────────────────────────────────────────────
        "Outcome":              "outcome",
    }

    df = df.rename(columns={k: v for k, v in ieg_map.items() if k in df.columns})

    # Ensure all required columns exist (empty if not in source)
    required = [
        "project_id", "project_name", "countryname", "regionname",
        "outcome", "quality_at_entry", "quality_of_supervision",
        "borrower_performance", "me_quality",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
            log.warning(f"  Column '{col}' not found in source — set to NaN")

    return df


def _fetch_projects_api(max_projects: int = None) -> pd.DataFrame:
    """Paginate WB Projects API. Returns metadata only (no IEG ratings)."""
    rows_per_page = 500
    all_projects, start = [], 0
    while True:
        params = {
            "format": "json", "rows": rows_per_page, "start": start,
            "fl": "id,project_name,countryname,regionname,boardapprovaldate,"
                  "closingdate,sector1,totalcommamt,status,prodline",
        }
        resp = requests.get(WB_PROJECTS_API, params=params, timeout=30)
        resp.raise_for_status()
        batch = list(resp.json().get("projects", {}).values())
        if not batch:
            break
        all_projects.extend(batch)
        if max_projects and len(all_projects) >= max_projects:
            break
        if len(batch) < rows_per_page:
            break
        start += rows_per_page
        time.sleep(0.5)
    return pd.DataFrame(all_projects[:max_projects] if max_projects else all_projects)


def _cache(df: pd.DataFrame) -> None:
    """Cache fetched data locally for offline reuse."""
    try:
        os.makedirs(os.path.dirname(LOCAL_DATA_PATH), exist_ok=True)
        df.to_csv(LOCAL_DATA_PATH, index=False)
        log.info(f"  Cached to {LOCAL_DATA_PATH}")
    except Exception as e:
        log.warning(f"  Cache write failed: {e}")


def get_fcs_countries() -> list:
    """Return Fragile & Conflict States country codes (WB classification)."""
    try:
        import wbgapi as wb
        fcs = wb.economy.list(q="fragile")
        return [c["id"] for c in fcs] if fcs else _fcs_static()
    except Exception:
        return _fcs_static()


def _fcs_static() -> list:
    """Static FCS fallback (WB 2024)."""
    return [
        "AFG","CAF","TCD","COD","ETH","GNB","HTI","IRQ","LBY","MLI",
        "MOZ","MMR","NER","NGA","PAK","PNG","SOM","SSD","SDN","SYR",
        "VEN","YEM","ZWE",
    ]
