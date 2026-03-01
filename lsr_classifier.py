"""
lsr_classifier.py
=================
Law of Symbiotic Resilience — Unified Classification Schema

THE SOLE INTELLECTUAL CONTRIBUTION OF THIS PIPELINE.
Classifies projects from four independent institutions onto the same theoretical scale.

LSR_Φ = 0.35×QaE + 0.25×QoS + 0.20×M&E + 0.20×BorrowerPerf
    Symbiotic : LSR_Φ ≥ 4.0
    Extractive: LSR_Φ < 4.0

Supports: World Bank IEG | ADB IED (2018, 2020-2024) | AidData GCDF

Author: Steven Brandon | Griffith University / QUT
SSRN: 7801909
"""
import pandas as pd
import numpy as np
import logging
log = logging.getLogger(__name__)

# ── Rating scale mappings ────────────────────────────────────────────────────────────
WB_6PT = {
    "Highly Satisfactory":6,"Satisfactory":5,"Moderately Satisfactory":4,
    "Moderately Unsatisfactory":3,"Unsatisfactory":2,"Highly Unsatisfactory":1,
    "Not Rated":np.nan,"Not Applicable":np.nan,
}
WB_4PT = {"High":4,"Substantial":3,"Modest":2,"Negligible":1,"Not Rated":np.nan}

ADB_2018_EFFECTIVENESS = {
    "Highly effective": 6, "Effective": 5, "Less than effective": 3,
    "Ineffective": 1, "Not rated": np.nan,
}
ADB_2018_SUSTAINABILITY = {
    "Most likely sustainable": 6, "Likely sustainable": 5,
    "Likely to be sustainable": 5, "Less than likely sustainable": 3,
    "Unlikely to be sustainable": 1, "Not rated": np.nan,
}
ADB_2018_RELEVANCE = {
    "Highly relevant": 6, "Relevant": 5, "Less than relevant": 3,
    "Irrelevant": 1, "Not rated": np.nan,
}

ADB_RATING = {
    "Highly Successful":6,"Successful":5,"Less than Successful":3,
    "Partly Successful":3,"Unsuccessful":1,
    "highly successful":6,"successful":5,"less than successful":3,
    "partly successful":3,"unsuccessful":1,"unsatisfactory ":1,
    "Highly Satisfactory":6,"Satisfactory":5,"Less than Satisfactory":3,
    "highly satisfactory":6,"satisfactory":5,"less than satisfactory":3,
    "less than statisfactory":3,"unsatisfactory":1,
    "-":np.nan,"na":np.nan,"nr":np.nan,"NR":np.nan,0:np.nan,"  ":np.nan,"…":np.nan,
}

AIDDATA_INTENT = {"Development":5.0,"Mixed":3.5,"Representational":2.0,"Commercial":2.0}
AIDDATA_FLOW   = {"ODA-like":5.5,"Vague (Official Finance)":3.5,"OOF-like":2.0}
AIDDATA_CONC   = {"Yes":5.0,"No":2.0}

WEIGHTS = {
    "quality_at_entry":0.35,"quality_of_supervision":0.25,
    "me_quality":0.20,"borrower_performance":0.20,
}
SYMBIOTIC_THRESHOLD = 4.0
SUCCESS_THRESHOLD   = 4.0


def apply_lsr_classification(df: pd.DataFrame, institution: str = "wb") -> pd.DataFrame:
    """
    Classify projects from any supported institution.

    Args:
        df:          DataFrame standardised by institution-specific prepare_* call,
                     OR raw WB IEG format (backward compatible)
        institution: "wb" | "adb" | "aiddata"

    Returns:
        df with columns: lsr_phi, lsr_class, T, success, hdi_group, decade
    """
    log.info(f"  Classifying {len(df):,} [{institution.upper()}]...")
    df = df.copy()

    if institution == "wb":
        df = _prep_wb(df)
    elif institution == "adb":
        df = _prep_adb(df)
    elif institution == "aiddata":
        df = _prep_aiddata(df)

    # LSR_Φ — same formula for all
    comps = {
        "quality_at_entry_n":      WEIGHTS["quality_at_entry"],
        "quality_of_supervision_n":WEIGHTS["quality_of_supervision"],
        "me_quality_n6":           WEIGHTS["me_quality"],
        "borrower_performance_n":  WEIGHTS["borrower_performance"],
    }
    avail  = {c: w for c, w in comps.items() if c in df.columns}
    if avail:
        tw = sum(avail.values())
        df["lsr_phi"] = sum(df[c].astype(float) * (w/tw) for c, w in avail.items())
    else:
        log.warning("  No rating columns found — lsr_phi = NaN")
        df["lsr_phi"] = np.nan

    df["lsr_class"] = np.where(df["lsr_phi"] >= SYMBIOTIC_THRESHOLD, "symbiotic",
                      np.where(df["lsr_phi"].notna(), "extractive", None))
    df["T"] = (df["lsr_class"] == "symbiotic").astype(int)

    if "outcome_n" in df.columns:
        df["success"] = np.where(df["outcome_n"].astype(float) >= SUCCESS_THRESHOLD, 1.0,
                        np.where(df["outcome_n"].notna(), 0.0, np.nan))

    df = _hdi(df)
    df = _decade(df)
    _log(df)
    return df


def _prep_wb(df):
    for col in ["quality_at_entry","quality_of_supervision","bank_performance",
                "borrower_performance","outcome"]:
        if col in df.columns:
            df[col+"_n"] = df[col].map(WB_6PT)
    if "me_quality" in df.columns:
        df["me_quality_n"]  = df["me_quality"].map(WB_4PT)
        df["me_quality_n6"] = (df["me_quality_n"] / 4) * 6
    if "outcome" in df.columns and "outcome_n" not in df.columns:
        df["outcome_n"] = df["outcome"].map(WB_6PT)
    df["_institution"] = "World Bank IEG"
    return df


def _prep_adb(df):
    rmap = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if   "ied overall"        in cl: rmap[col] = "ied_outcome"
        elif "pper overall"       in cl: rmap[col] = "pper_outcome"
        elif "latest overall"     in cl: rmap[col] = "latest_outcome"
        elif "ied borrower"       in cl: rmap[col] = "ied_bp"
        elif "pper borrower"      in cl: rmap[col] = "pper_bp"
        elif "ied adb"            in cl: rmap[col] = "ied_qos"
        elif "pcr quality"        in cl: rmap[col] = "pcr_quality"
        elif "country name"       in cl: rmap[col] = "countryname"
        elif "country code"       in cl: rmap[col] = "country_code"
        elif "country group"      in cl: rmap[col] = "regionname"
        elif "regional group"     in cl and "regionname" not in rmap.values(): rmap[col] = "regionname"
        elif "country name"  ==   cl: rmap[col] = "countryname"
        elif "primary sector" in cl and "2014" in cl and not cl.endswith("_3") and "sector1" not in rmap.values():
            rmap[col] = "sector1"
        elif "sector (2009)" in cl and "sector1" not in rmap.values():
            rmap[col] = "sector1"
        elif "approved amount"    in cl: rmap[col] = "totalcommamt"
        elif "approval year" in cl: rmap[col] = "boardapprovaldate"
        elif "project no."        in cl: rmap[col] = "project_id"
        elif "project name"       in cl: rmap[col] = "project_name"
        elif cl == "sids":               rmap[col] = "is_SIDS"
        elif cl == "fcas":               rmap[col] = "is_FCAS"
    df = df.rename(columns=rmap)

    for src in ["ied_outcome","pper_outcome","latest_outcome"]:
        if src in df.columns:
            df["outcome"] = df[src]; break
    if "pcr_quality"  in df.columns: df["quality_at_entry"]       = df["pcr_quality"]
    if "ied_qos"      in df.columns: df["quality_of_supervision"]  = df["ied_qos"]
    for src in ["ied_bp","pper_bp"]:
        if src in df.columns: df["borrower_performance"] = df[src]; break
    # 2018 dataset uses component-level ratings (effectiveness/sustainability/relevance)
    # instead of combined quality scores — apply specific rating scales
    is_2018 = "_aer_year" in df.columns and (df["_aer_year"] == 2018).all()
    if is_2018:
        if "PCR effectiveness" in df.columns:
            df["quality_of_supervision"] = df["PCR effectiveness"]
            df["quality_of_supervision_n"] = df["PCR effectiveness"].map(ADB_2018_EFFECTIVENESS)
        if "PCR sustainability" in df.columns:
            df["me_quality"] = df["PCR sustainability"]
            df["me_quality_n"] = df["PCR sustainability"].map(ADB_2018_SUSTAINABILITY)
            df["me_quality_n6"] = df["me_quality_n"]
        if "PCR relevance" in df.columns:
            df["quality_at_entry"] = df["PCR relevance"]
            df["quality_at_entry_n"] = df["PCR relevance"].map(ADB_2018_RELEVANCE)
    else:
        df["me_quality"] = df.get("quality_at_entry", np.nan)
        for col in ["outcome","quality_at_entry","quality_of_supervision",
                    "borrower_performance","me_quality"]:
            if col in df.columns:
                df[col+"_n"] = df[col].map(ADB_RATING)
        df["me_quality_n6"] = df.get("me_quality_n", pd.Series(np.nan, index=df.index))

    df["_institution"] = "ADB IED"
    return df


def _prep_aiddata(df):
    df = df.rename(columns={
        "Recipient":"countryname","Recipient Region":"regionname",
        "Sector Name":"sector1","Amount (Constant USD2017)":"totalcommamt",
        "Commitment Year":"boardapprovaldate","Title":"project_name",
        "AidData TUFF Project ID":"project_id",
    })
    df["quality_at_entry_n"]       = df["Intent"].map(AIDDATA_INTENT) if "Intent" in df.columns else 3.0
    df["quality_of_supervision_n"] = df["Flow Class"].map(AIDDATA_FLOW) if "Flow Class" in df.columns else 3.5
    df["borrower_performance_n"]   = df["Concessional"].map(AIDDATA_CONC) if "Concessional" in df.columns else 3.0
    if "Grant Element" in df.columns:
        g = df["Grant Element"].clip(0,100).fillna(50)/100
        df["me_quality_n6"] = 1.0 + g * 5.0
    else:
        df["me_quality_n6"] = 3.0
    df["outcome_n"] = np.where(df["Status"]=="Completion", 5.0,
                      np.where(df["Status"].isin(["Cancelled","Suspended"]), 1.0, np.nan))
    df["_institution"] = "AidData GCDF"
    return df


def _hdi(df):
    high = ["Europe and Central Asia","Latin America & Caribbean","East Asia and Pacific"]
    low  = ["Sub-Saharan Africa","South Asia","Africa"]
    if "regionname" in df.columns:
        df["hdi_group"] = np.select(
            [df["regionname"].isin(high), df["regionname"].isin(low)],
            ["High","Low"], default="Medium")
    else:
        df["hdi_group"] = "Medium"
    return df


def _decade(df):
    yr = None
    # Prefer actual approval year over AER publication year
    for col in ["boardapprovaldate", "approval_year", "Approval year"]:
        if col in df.columns:
            try:
                yr = pd.to_numeric(df[col], errors="coerce")
                if yr.isna().all():
                    yr = pd.to_datetime(df[col], errors="coerce").dt.year
                break
            except Exception:
                continue
    if yr is not None and not isinstance(yr, type(None)):
        df["approval_year"] = yr
        df["decade"] = ((yr // 10) * 10).fillna(2010)
    else:
        df["decade"] = 2010
    return df


def _log(df):
    inst = df.get("_institution", pd.Series(["?"])).iloc[0] if len(df) else "?"
    cls  = df["lsr_class"].notna().sum()
    sym  = (df["lsr_class"]=="symbiotic").sum()
    ext  = (df["lsr_class"]=="extractive").sum()
    log.info(f"  [{inst}] {cls:,} classified | sym={sym:,} ext={ext:,}")
    if "success" in df.columns and cls > 0:
        ss = df.loc[df["lsr_class"]=="symbiotic","success"].mean()
        es = df.loc[df["lsr_class"]=="extractive","success"].mean()
        if 0 < es < 1 and ss > 0:
            orv = (ss/(1-ss))/(es/(1-es))
            log.info(f"  [{inst}] sym={ss*100:.1f}% ext={es*100:.1f}% OR={orv:.2f}x")
