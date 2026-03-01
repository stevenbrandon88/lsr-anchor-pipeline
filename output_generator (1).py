"""
output_generator.py
===================
Generates the final Excel workbook and summary files.
Handles multi-institutional results.
"""
import pandas as pd, numpy as np, logging, os
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
log = logging.getLogger(__name__)

GREEN  = "2ea043"; BLUE = "58a6ff"; ORANGE = "f8c27d"; RED = "f85149"
BG     = "0d1117"; BG2  = "161b22"


def generate_outputs(classified_datasets: dict, cross: dict, ts: str) -> None:
    """Write Excel workbook with all results."""
    path = f"outputs/lsr_results_{ts}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _sheet_summary(writer, classified_datasets, cross)
        _sheet_comparison(writer, cross)
        for label, df in classified_datasets.items():
            safe = label.replace(" ","_").replace("/","")[:25]
            _sheet_institution(writer, label, df, safe)
        _sheet_methodology(writer)
    log.info(f"  Excel saved: {path}")

    # Plain text summary
    txt_path = f"outputs/lsr_summary_{ts}.txt"
    _write_summary_txt(txt_path, classified_datasets, cross)


def _sheet_summary(writer, datasets, cross):
    n_conf  = cross.get("n_confirming", 0)
    n_total = cross.get("n_total", 0)
    rows = [
        ["LSR ANCHOR PIPELINE — RESULTS SUMMARY", ""],
        ["", ""],
        ["HEADLINE FINDING", f"{n_conf}/{n_total} independent datasets confirm LSR effect"],
        ["", ""],
    ]
    for label, df in datasets.items():
        cls = df["lsr_class"].notna().sum()
        sym = (df["lsr_class"]=="symbiotic").sum()
        ext = (df["lsr_class"]=="extractive").sum()
        rows.append([f"[{label}] Projects classified", f"{cls:,} ({sym:,} symbiotic, {ext:,} extractive)"])
        if "success" in df.columns and cls > 0:
            s1 = df.loc[df["lsr_class"]=="symbiotic","success"].mean()
            s0 = df.loc[df["lsr_class"]=="extractive","success"].mean()
            rows.append([f"[{label}] Symbiotic success rate", f"{s1*100:.1f}%"])
            rows.append([f"[{label}] Extractive success rate", f"{s0*100:.1f}%"])
            if 0 < s0 < 1:
                orv = (s1/(1-s1))/(s0/(1-s0))
                rows.append([f"[{label}] Odds Ratio", f"{orv:.2f}×"])
        rows.append(["",""])

    inst_results = cross.get("results", {})
    for label, r in inst_results.items():
        rows += [
            [f"[{label}] ATE (DoWhy causal)", f"+{r.get('ate',np.nan)*100:.1f}pp"],
            [f"[{label}] Spec curve specs", f"{r.get('n_specs',0):,}"],
            [f"[{label}] % Confirming", f"{r.get('pct_confirming',np.nan):.1f}%"],
            [f"[{label}] Refutations", f"{r.get('refutations_passed',0)}/3 PASS"],
            ["", ""],
        ]

    pd.DataFrame(rows, columns=["Metric", "Value"]).to_excel(
        writer, sheet_name="Summary", index=False)


def _sheet_comparison(writer, cross):
    comp = cross.get("comparison")
    if comp is not None and len(comp) > 0:
        comp.to_excel(writer, sheet_name="Cross-Institutional", index=False)


def _sheet_institution(writer, label, df, safe_label):
    # Sample (up to 1000)
    sample = df[["project_id","project_name","countryname","regionname",
                 "lsr_phi","lsr_class","T","success","decade","hdi_group"]
                ].dropna(subset=["lsr_class"]).head(1000)
    sample.to_excel(writer, sheet_name=f"Projects_{safe_label}", index=False)


def _sheet_methodology(writer):
    rows = [
        ["MODULE", "TOOL", "DEVELOPED BY", "PURPOSE"],
        ["Specification curve", "RobustiPy", "Lachlan Kochanski (2022)", "All control combinations"],
        ["Causal identification","DoWhy","Microsoft Research","DAG + refutation tests"],
        ["Heterogeneous effects","EconML","Microsoft Research","CATE by HDI/region"],
        ["","","",""],
        ["DATA SOURCES", "INSTITUTION", "EVALUATOR", "INDEPENDENCE"],
        ["World Bank IEG","World Bank","IEG division","Fully independent from operations"],
        ["ADB IED","Asian Dev Bank","IED division","Fully independent from operations"],
        ["AidData GCDF","William & Mary","AidData team","Independent researchers"],
        ["","","",""],
        ["RESEARCHER CONTRIBUTION","lsr_classifier.py ONLY","",""],
        ["Classification schema","LSR_Phi formula","Brandon (2025)","SSRN 7801909"],
    ]
    pd.DataFrame(rows[1:], columns=rows[0]).to_excel(
        writer, sheet_name="Methodology", index=False)


def _write_summary_txt(path, datasets, cross):
    n_conf  = cross.get("n_confirming", 0)
    n_total = cross.get("n_total", 0)
    lines = [
        "=" * 65,
        "LSR ANCHOR PIPELINE — SUMMARY",
        "=" * 65,
        "",
        f"HEADLINE: {n_conf}/{n_total} independent datasets confirm LSR effect",
        "",
    ]
    inst_results = cross.get("results", {})
    for label, r in inst_results.items():
        lines += [
            f"── {label} ──────────────────────────────────",
            f"  n={r.get('n',0):,} | sym={r.get('n_symbiotic',0):,} | ext={r.get('n_extractive',0):,}",
            f"  Symbiotic success: {r.get('sym_success',np.nan)*100:.1f}%",
            f"  Extractive success: {r.get('ext_success',np.nan)*100:.1f}%",
            f"  Odds Ratio: {r.get('odds_ratio',np.nan):.2f}×",
            f"  ATE (DoWhy): +{r.get('ate',np.nan)*100:.1f}pp",
            f"  Spec curve: {r.get('n_specs',0):,} specs | {r.get('pct_confirming',np.nan):.1f}% confirming",
            f"  Refutations: {r.get('refutations_passed',0)}/3 PASS",
            f"  CONFIRMS LSR: {'YES' if r.get('confirms_lsr') else 'NO'}",
            "",
        ]
    lines += [
        "=" * 65,
        "Reproduce: python run_pipeline.py",
        "Data: World Bank IEG + ADB IED + AidData GCDF",
        "Methods: RobustiPy + DoWhy (Microsoft) + EconML (Microsoft)",
        "Author: Steven Brandon | Griffith University / QUT",
        "SSRN: 7801909",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"  Summary text: {path}")
