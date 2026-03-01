"""
institution_fetchers.py
=======================
Data fetchers for all four institutions.
Each returns a raw DataFrame ready for lsr_classifier.

Institutions:
    WB  — World Bank IEG (live API → local cache)
    ADB — ADB IED Annual Evaluation Reports 2018 + 2020-2024 (local files)
    AID — AidData Global Chinese Development Finance v2.0 (local file)

For each institution, the fetcher:
    1. Tries public download (where available)
    2. Falls back to data/{institution}/ folder
    3. Logs data source and timestamp for audit trail
"""
import os, time, logging, requests
import pandas as pd, numpy as np
from io import StringIO
from glob import glob

log = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ══════════════════════════════════════════════════════════════════════════
#  INSTITUTION 1: WORLD BANK IEG
# ══════════════════════════════════════════════════════════════════════════

WB_IEG_URLS = [
    "https://ieg.worldbankgroup.org/sites/default/files/Data/reports/ieg_world_bank_project_performance_ratings.csv",
    "https://datacatalogfiles.worldbank.org/ddh-published/0037394/DR0045406/ieg_world_bank_project_performance_ratings.csv",
]
WB_LOCAL = os.path.join(DATA_DIR, "wb", "ieg_ratings.csv")
WB_PROJECTS_API = "https://search.worldbank.org/api/v2/projects"


def fetch_wb_ieg(max_rows=None, force_local=False) -> pd.DataFrame:
    """World Bank IEG project performance ratings."""
    if not force_local:
        for url in WB_IEG_URLS:
            try:
                log.info(f"  [WB] Fetching: {url[:60]}")
                r = requests.get(url, timeout=90)
                r.raise_for_status()
                df = pd.read_csv(StringIO(r.text))
                log.info(f"  [WB] [OK] {len(df):,} rows from live API")
                df = _standardise_wb(df)
                if max_rows: df = df.head(max_rows)
                _save(df, WB_LOCAL)
                return df
            except Exception as e:
                log.warning(f"  [WB] URL failed: {e}")
        try:
            log.info("  [WB] Trying Projects API...")
            df = _wb_projects_api(max_rows)
            return _standardise_wb(df)
        except Exception as e:
            log.warning(f"  [WB] Projects API failed: {e}")

    if os.path.exists(WB_LOCAL):
        df = pd.read_csv(WB_LOCAL)
        if max_rows: df = df.head(max_rows)
        log.info(f"  [WB] [OK] {len(df):,} rows from local cache")
        return df

    raise RuntimeError(
        "World Bank IEG data not available.\n"
        "  A: Connect to internet and rerun\n"
        "  B: Download from https://ieg.worldbankgroup.org/evaluations/data\n"
        "     Save as: data/wb/ieg_ratings.csv"
    )


def _standardise_wb(df):
    """Map actual WB IEG CSV column names (abbreviated) to internal names."""
    col_map = {
        # Short names (actual IEG CSV format)
        "QaE":                  "quality_at_entry",
        "QoS":                  "quality_of_supervision",
        "Borr Perf":            "borrower_performance",
        "M&E Qual":             "me_quality",
        "Bank Perf":            "bank_performance",
        "Proj ID":              "project_id",
        "Proj Name":            "project_name",
        "Proj Cost":            "totalcommamt",
        "Country":              "countryname",
        "Region":               "regionname",
        "Approval FY":          "boardapprovaldate",
        "Exit FY":              "closingdate",
        "Commit Amt (USD)":     "totalcommamt",
        "Sector (Code)":        "sector1",
        "Lending Instr":        "lending_instrument",
        "Outcome":              "outcome",
        "Risk to Dev Outcome":  "risk_to_dev_outcome",
        # Long names (alternate download format)
        "Quality at Entry":          "quality_at_entry",
        "Quality of Supervision":    "quality_of_supervision",
        "Borrower Performance":      "borrower_performance",
        "M&E Quality":               "me_quality",
        "Bank Performance":          "bank_performance",
        "Project ID":                "project_id",
        "Project Name":              "project_name",
        "Project Cost (USD)":        "totalcommamt",
        "Board Approval Date":       "boardapprovaldate",
        "Closing Date":              "closingdate",
        "Sector":                    "sector1",
        "Sector Board":              "sector1",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    for req in ["project_id","countryname","regionname","outcome",
                "quality_at_entry","quality_of_supervision","borrower_performance"]:
        if req not in df.columns:
            df[req] = np.nan
    return df


def _wb_projects_api(max_rows=None):
    """Paginate WB Projects API (metadata only, no IEG ratings)."""
    rows = 500; all_p = []; start = 0
    while True:
        r = requests.get(WB_PROJECTS_API, timeout=30, params={
            "format":"json","rows":rows,"start":start,
            "fl":"id,project_name,countryname,regionname,boardapprovaldate,closingdate,sector1,totalcommamt",
        })
        r.raise_for_status()
        batch = list(r.json().get("projects",{}).values())
        if not batch: break
        all_p.extend(batch)
        if max_rows and len(all_p) >= max_rows: break
        if len(batch) < rows: break
        start += rows; time.sleep(0.5)
    return pd.DataFrame(all_p[:max_rows] if max_rows else all_p)


# ══════════════════════════════════════════════════════════════════════════
#  INSTITUTION 2: ADB IED (Annual Evaluation Reports 2018 + 2020-2024)
# ══════════════════════════════════════════════════════════════════════════

ADB_FILES = {
    2018: os.path.join(DATA_DIR, "adb", "adb-success-rates-database-2018.xlsx"),
    2020: os.path.join(DATA_DIR, "adb", "AER-2020-Success-Rates.xlsx"),
    2021: os.path.join(DATA_DIR, "adb", "AER-2021-Success-Database.xlsx"),
    2022: os.path.join(DATA_DIR, "adb", "AER-2022-Success-Database.xlsx"),
    2023: os.path.join(DATA_DIR, "adb", "aer-2023_success-rate-database.xlsx"),
    2024: os.path.join(DATA_DIR, "adb", "aer-2024-success-rates-database.xlsx"),
}
ADB_SHEETS = {2018: "Data", 2020: "AER data", 2021: "AER data",
              2022: "AER data", 2023: "AER data", 2024: "AER data"}

# 2018 uses different column names for component ratings
ADB_2018_REMAP = {
    "PCR effectiveness":  "quality_of_supervision",   # best available proxy for QoS
    "PCR sustainability": "me_quality",                # proxy for M&E / sustainability
    "PCR relevance":      "quality_at_entry",          # proxy for QaE
    # No borrower performance column in 2018 — weight renormalised automatically
}


def fetch_adb_ied(years=None, max_rows=None) -> pd.DataFrame:
    """
    Load ADB IED Annual Evaluation Report data.
    Combines multiple annual cohorts for maximum sample size.

    Files needed in data/adb/:
        adb-success-rates-database-2018.xlsx  (2018 methodology)
        AER-2020-Success-Rates.xlsx
        AER-2021-Success-Database.xlsx
        AER-2022-Success-Database.xlsx
        aer-2023_success-rate-database.xlsx
        aer-2024-success-rates-database.xlsx

    Download from: https://www.adb.org/what-we-do/evaluation/data

    Args:
        years: list of years to load (default: all available)
    """
    years = years or list(ADB_FILES.keys())
    frames = []

    for yr in years:
        path = ADB_FILES[yr]
        if not os.path.exists(path):
            log.warning(f"  [ADB] {yr} file not found: {path}")
            continue
        try:
            sheet = ADB_SHEETS[yr]
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            df["_aer_year"] = yr
            frames.append(df)
            log.info(f"  [ADB] [OK] {yr}: {len(df):,} rows")
        except Exception as e:
            log.warning(f"  [ADB] Failed to load {yr}: {e}")

    if not frames:
        raise RuntimeError(
            "No ADB IED files found.\n"
            "  Download from: https://www.adb.org/what-we-do/evaluation/data\n"
            "  Place in: data/adb/"
        )

    df = pd.concat(frames, ignore_index=True)
    log.info(f"  [ADB] Combined: {len(df):,} project-year records from {len(frames)} cohorts")

    # Deduplicate on project ID (keep latest rating)
    id_col = next((c for c in df.columns if "project no" in str(c).lower()), None)
    if id_col:
        df = df.sort_values("_aer_year", ascending=False).drop_duplicates(
            subset=[id_col], keep="first")
        log.info(f"  [ADB] After dedup: {len(df):,} unique projects")

    if max_rows: df = df.head(max_rows)
    return df


# ══════════════════════════════════════════════════════════════════════════
#  INSTITUTION 3: AIDDATA GLOBAL CHINESE DEVELOPMENT FINANCE v2.0
# ══════════════════════════════════════════════════════════════════════════

AIDDATA_LOCAL = os.path.join(DATA_DIR, "aiddata",
    "AidDatasGlobalChineseDevelopmentFinanceDataset_v2_0.xlsx")


def fetch_aiddata(filter_resolved=True, filter_recommended=True,
                  max_rows=None) -> pd.DataFrame:
    """
    Load AidData Global Chinese Development Finance Dataset v2.0.

    File needed in data/aiddata/:
        AidDatasGlobalChineseDevelopmentFinanceDataset_v2_0.xlsx

    Download from:
        https://www.aiddata.org/data/aiddatas-global-chinese-development-finance-dataset-version-2-0

    Args:
        filter_resolved:    Keep only Completion/Cancelled/Suspended (excludes Pipeline)
        filter_recommended: Keep only Recommended For Aggregates == Yes
    """
    paths = [AIDDATA_LOCAL] + glob(os.path.join(DATA_DIR, "aiddata", "*.xlsx"))
    loaded = False
    for path in paths:
        if os.path.exists(path):
            try:
                log.info(f"  [AID] Loading: {os.path.basename(path)}")
                df = pd.read_excel(path, sheet_name="Global_CDF2.0", engine="openpyxl")
                loaded = True
                log.info(f"  [AID] [OK] {len(df):,} rows loaded")
                break
            except Exception as e:
                log.warning(f"  [AID] Failed: {e}")

        # Also try CSV fallback
    if not loaded:
        csv_df = _load_aiddata_csv(max_rows)
        if csv_df is not None:
            return csv_df

    if not loaded:
        raise RuntimeError(
            "AidData GCDF file not found.\n"
            "  Download from: https://www.aiddata.org/data/\n"
            "      aiddatas-global-chinese-development-finance-dataset-version-2-0\n"
            "  Place xlsx in: data/aiddata/"
        )

    if filter_recommended and "Recommended For Aggregates" in df.columns:
        df = df[df["Recommended For Aggregates"] == "Yes"]
        log.info(f"  [AID] After quality filter: {len(df):,}")

    if filter_resolved and "Status" in df.columns:
        df = df[df["Status"].isin(["Completion","Cancelled","Suspended"])]
        log.info(f"  [AID] After resolved filter: {len(df):,}")

    if max_rows: df = df.head(max_rows)
    return df


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _save(df, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        log.info(f"  Cached to {path}")
    except Exception as e:
        log.warning(f"  Cache write failed: {e}")


def get_fcs_countries():
    """WB Fragile & Conflict States country codes."""
    try:
        import wbgapi as wb
        return [c["id"] for c in wb.economy.list(q="fragile")]
    except Exception:
        return ["AFG","CAF","TCD","COD","ETH","GNB","HTI","IRQ","LBY","MLI",
                "MOZ","MMR","NER","NGA","PAK","PNG","SOM","SSD","SDN","SYR","YEM","ZWE"]


AIDDATA_CSV_LOCAL = os.path.join(DATA_DIR, "aiddata", "aiddata_gcdf.csv")  # pre-classified CSV fallback


def _load_aiddata_csv(max_rows=None):
    """Load from pre-classified CSV if xlsx not available."""
    if os.path.exists(AIDDATA_CSV_LOCAL):
        df = pd.read_csv(AIDDATA_CSV_LOCAL)
        if max_rows: df = df.head(max_rows)
        log.info(f"  [AID] [OK] {len(df):,} rows from CSV cache")
        return df
    return None
