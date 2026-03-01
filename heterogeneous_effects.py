"""
heterogeneous_effects.py - EconML Causal Forest DML wrapper.
API: discrete_outcome=True, discrete_treatment=True (NOT model_y/model_t classifiers)
"""
import numpy as np, pandas as pd, logging, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)
FEATURES = ["decade","region_enc","sector_enc","totalcommamt_log","is_SIDS","is_FCAS","is_fcs"]

def run_econml(df: pd.DataFrame) -> dict:
    try:
        from econml.dml import CausalForestDML
    except ImportError:
        log.error("EconML not installed: pip install econml"); return _empty()
    avail = [c for c in FEATURES if c in df.columns]
    group_cols = [c for c in ["regionname","sector1","hdi_group"] if c in df.columns]
    df_e = df[["success","T"]+avail+group_cols].dropna(subset=["success","T"]).copy()
    for col in avail:
        df_e[col] = pd.to_numeric(df_e[col],errors="coerce").fillna(df_e[col].median() if df_e[col].notna().any() else 0)
    if len(df_e) < 100:
        log.warning(f"  EconML: only {len(df_e)} rows — skipping"); return _empty()
    Y=df_e["success"].values.astype(float); T=df_e["T"].values.astype(float)
    X=df_e[avail].values.astype(float) if avail else np.zeros((len(df_e),1))
    log.info(f"  EconML: n={len(df_e):,} | features={avail}")
    cf = CausalForestDML(discrete_outcome=True,discrete_treatment=True,
                         n_estimators=300,min_samples_leaf=10,random_state=42,n_jobs=-1)
    cf.fit(Y, T, X=X)
    cate = cf.effect(X); df_e["cate"] = cate
    mean_cate = float(cate.mean())
    log.info(f"  EconML mean CATE: +{mean_cate*100:.1f}pp")
    region_effects = {}
    for col in group_cols:
        for grp, sub in df_e.groupby(col):
            if len(sub) >= 5:
                region_effects[f"{col}:{grp}"] = {"mean_cate":float(sub["cate"].mean()),"n":len(sub)}
    fig_path = _plot(region_effects, mean_cate)
    return {"mean_cate":mean_cate,"std_cate":float(cate.std()),"hdi_effects":{},"region_effects":region_effects,"cate_series":df_e["cate"],"figure_path":fig_path}

def _plot(effects, mean_cate):
    if not effects: return None
    labels = sorted(effects.keys(), key=lambda k: effects[k]["mean_cate"])[:20]
    vals = [effects[l]["mean_cate"]*100 for l in labels]
    fig, ax = plt.subplots(figsize=(10,max(5,len(labels)*0.4+2)))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    ax.spines[:].set_color("#30363d"); ax.tick_params(colors="#8b949e",labelsize=7)
    colors=["#2ea043" if v==max(vals) else "#58a6ff" if v>0 else "#f85149" for v in vals]
    ax.barh(labels,vals,color=colors,alpha=0.85,edgecolor="#30363d")
    ax.axvline(mean_cate*100,color="#f8c27d",lw=2,ls="--",label=f"Mean {mean_cate*100:.1f}pp")
    ax.set_title("Heterogeneous Treatment Effects (EconML Causal Forest)",color="#c9d1d9",fontweight="bold")
    ax.set_xlabel("CATE (pp)",color="#8b949e"); ax.legend(framealpha=0.3,labelcolor="#c9d1d9")
    os.makedirs("outputs",exist_ok=True); path="outputs/hte_effects.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight",facecolor="#0d1117"); plt.close()
    return path

def _empty():
    return {"mean_cate":np.nan,"std_cate":np.nan,"hdi_effects":{},"region_effects":{},"cate_series":pd.Series(dtype=float),"figure_path":None}
