"""
run_pipeline.py
===============
LSR Anchor Pipeline — Multi-Institutional Entry Point

USAGE:
    python run_pipeline.py                    # all available institutions
    python run_pipeline.py --institutions wb adb aid  # specific subset
    python run_pipeline.py --max 500          # limit rows per institution (test)
    python run_pipeline.py --local            # use local cached files only

OUTPUT:
    outputs/lsr_results_[timestamp].xlsx      # 8-sheet workbook
    outputs/cross_institutional.png           # 3-panel comparison figure
    outputs/spec_curve_[institution].png      # per-institution spec curves
    outputs/hte_effects_[institution].png     # per-institution HTE
    logs/pipeline_[timestamp].log             # full audit trail

Author: Steven Brandon (s.brandon@griffith.edu.au)
Framework: Law of Symbiotic Resilience — SSRN 7801909
"""

import os, sys, logging, datetime, warnings, argparse
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

os.makedirs("outputs", exist_ok=True)
os.makedirs("logs",    exist_ok=True)
os.makedirs("data/wb", exist_ok=True)
os.makedirs("data/adb", exist_ok=True)
os.makedirs("data/aiddata", exist_ok=True)

ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log = logging.getLogger()
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
for h in [logging.FileHandler(f"logs/pipeline_{ts}.log"), logging.StreamHandler(sys.stdout)]:
    h.setFormatter(fmt); log.addHandler(h)

from institution_fetchers import fetch_wb_ieg, fetch_adb_ied, fetch_aiddata
from lsr_classifier import apply_lsr_classification
from robustness_analysis import run_robustipy
from causal_analysis import run_dowhy
from heterogeneous_effects import run_econml
from cross_institutional import run_cross_institutional
from output_generator import generate_outputs


def main():
    parser = argparse.ArgumentParser(description="LSR Multi-Institutional Pipeline")
    parser.add_argument("--institutions", nargs="+", default=["wb","adb","aid"],
                        choices=["wb","adb","aid"],
                        help="Which institutions to run (default: all)")
    parser.add_argument("--max",   type=int, default=None, help="Max rows per institution")
    parser.add_argument("--local", action="store_true",    help="Use cached files only")
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("LSR ANCHOR PIPELINE — MULTI-INSTITUTIONAL VALIDATION")
    log.info(f"Run: {ts}")
    log.info(f"Institutions: {args.institutions}")
    log.info("=" * 65)

    classified_datasets = {}

    # ── INSTITUTION 1: World Bank IEG ─────────────────────────────────────────
    if "wb" in args.institutions:
        log.info("\n[WB] World Bank Independent Evaluation Group")
        try:
            df_raw = fetch_wb_ieg(max_rows=args.max, force_local=args.local)
            df_wb  = apply_lsr_classification(df_raw, institution="wb")
            classified_datasets["World Bank IEG"] = df_wb
            _log_headline("World Bank IEG", df_wb)
        except Exception as e:
            log.warning(f"  [WB] Skipped: {e}")

    # ── INSTITUTION 2: ADB IED ────────────────────────────────────────────────
    if "adb" in args.institutions:
        log.info("\n[ADB] Asian Development Bank Independent Evaluation Department")
        try:
            df_raw = fetch_adb_ied(max_rows=args.max)
            df_adb = apply_lsr_classification(df_raw, institution="adb")
            classified_datasets["ADB IED"] = df_adb
            _log_headline("ADB IED", df_adb)
        except Exception as e:
            log.warning(f"  [ADB] Skipped: {e}")

    # ── INSTITUTION 3: AidData GCDF ───────────────────────────────────────────
    if "aid" in args.institutions:
        log.info("\n[AID] AidData Global Chinese Development Finance v2.0")
        try:
            df_raw = fetch_aiddata(max_rows=args.max)
            df_aid = apply_lsr_classification(df_raw, institution="aiddata")
            classified_datasets["AidData GCDF"] = df_aid
            _log_headline("AidData GCDF", df_aid)
        except Exception as e:
            log.warning(f"  [AID] Skipped: {e}")

    if not classified_datasets:
        log.error("No institutions loaded. Check data files and try again.")
        sys.exit(1)

    log.info(f"\n{len(classified_datasets)} institution(s) loaded. Running validation...")

    # ── CROSS-INSTITUTIONAL COMPARISON ────────────────────────────────────────
    log.info("\n[CROSS] Running cross-institutional comparison...")
    cross = run_cross_institutional(classified_datasets)

    n_conf  = cross.get("n_confirming", 0)
    n_total = cross.get("n_total", 0)
    log.info(f"\n{'='*65}")
    log.info(f"CROSS-INSTITUTIONAL RESULT: {n_conf}/{n_total} datasets confirm LSR")
    log.info(f"{'='*65}")
    if cross.get("comparison") is not None:
        log.info(cross["comparison"].to_string(index=False))

    # ── GENERATE OUTPUTS ──────────────────────────────────────────────────────
    log.info("\nGenerating Excel workbook...")
    generate_outputs(classified_datasets, cross, ts)

    log.info(f"\nPipeline complete.")
    log.info(f"  outputs/lsr_results_{ts}.xlsx")
    log.info(f"  outputs/cross_institutional.png")
    log.info(f"  logs/pipeline_{ts}.log")


def _log_headline(label, df):
    cls = df["lsr_class"].notna().sum()
    sym = (df["lsr_class"]=="symbiotic").sum()
    ext = (df["lsr_class"]=="extractive").sum()
    if "success" in df.columns and cls > 0:
        s1 = df.loc[df["lsr_class"]=="symbiotic","success"].mean()
        s0 = df.loc[df["lsr_class"]=="extractive","success"].mean()
        if 0 < s0 < 1:
            orv = (s1/(1-s1))/(s0/(1-s0))
            log.info(f"  [{label}] {cls:,} classified | sym={s1*100:.1f}% ext={s0*100:.1f}% OR={orv:.2f}×")
        else:
            log.info(f"  [{label}] {cls:,} classified | sym={sym:,} ext={ext:,}")


if __name__ == "__main__":
    main()
