"""
robustness_analysis.py
======================
Specification Curve via RobustiPy.
API note: m.fit() → m.get_results() (NOT results = m.fit())
"""
import numpy as np, pandas as pd, logging, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)

CANDIDATE_CONTROLS = [
    "decade","region_enc","sector_enc","country_enc",
    "totalcommamt_log","is_SIDS","is_FCAS","is_fcs",
]

def run_robustipy(df: pd.DataFrame) -> dict:
    from robustipy.models import OLSRobust
    y_var = "success"; x_var = "T"

    # Ensure categoricals are encoded
    df_c = df.copy()
    for col, enc in [("regionname","region_enc"), ("sector1","sector_enc"), ("countryname","country_enc")]:
        if col in df_c.columns and enc not in df_c.columns:
            df_c[enc] = pd.Categorical(df_c[col]).codes.astype(float)
            df_c.loc[df_c[enc] < 0, enc] = np.nan
    if "totalcommamt" in df_c.columns and "totalcommamt_log" not in df_c.columns:
        df_c["totalcommamt_log"] = np.log1p(pd.to_numeric(df_c["totalcommamt"], errors="coerce").clip(lower=0))

    controls = [c for c in CANDIDATE_CONTROLS if c in df_c.columns]
    df_c = df_c[["success","T"] + controls].copy()
    for col in controls:
        df_c[col] = pd.to_numeric(df_c[col], errors="coerce")
    df_c = df_c.dropna(subset=["success","T"])
    # fill NaN controls with median to preserve sample size
    for col in controls:
        df_c[col] = df_c[col].fillna(df_c[col].median())

    if len(df_c) < 50:
        log.warning("  RobustiPy: <50 obs — skipping"); return _empty()

    log.info(f"  RobustiPy: n={len(df_c):,} | controls={controls}")
    try:
        m = OLSRobust(y=[y_var], x=[x_var], data=df_c)
        m.fit(controls=controls, draws=200, kfold=10, seed=42, oos_metric="pseudo-r2", n_cpu=1)
        r = m.get_results()
        ests = r.estimates.values.flatten(); pvs = r.p_values.values.flatten()
        n = len(ests); med = float(np.nanmedian(ests))
        confirm = float(np.mean((ests > 0) & (pvs < 0.05)) * 100)
        log.info(f"  RobustiPy: {n} specs | median={med:.4f} | confirming={confirm:.1f}%")
        return {"n_specs":n,"median_coef":med,"pct_positive":float(np.mean(ests>0)*100),
                "pct_significant":float(np.mean(pvs<0.05)*100),"pct_confirming":confirm,
                "estimates":ests,"p_values":pvs,"figure_path":_plot(ests,pvs,n,med,confirm)}
    except Exception as e:
        log.warning(f"  RobustiPy error: {e} — running statsmodels fallback")
        return _fallback(df_c, y_var, x_var, controls)


def _fallback(df_c, y_var, x_var, controls):
    """Manual spec curve via statsmodels — runs without robustipy."""
    import statsmodels.api as sm
    from itertools import combinations
    all_coefs, all_pvals = [], []
    for n in range(0, min(len(controls)+1, 6)):
        for combo in combinations(controls, n):
            cols_needed = [y_var, x_var] + list(combo)
            # all columns must exist in df_c
            missing = [c for c in cols_needed if c not in df_c.columns]
            if missing: continue
            sub = df_c[cols_needed].dropna()
            if len(sub) < 30: continue
            try:
                X_cols = [x_var] + list(combo)
                m = sm.OLS(sub[y_var], sm.add_constant(sub[X_cols])).fit(cov_type="HC3")
                all_coefs.append(m.params[x_var]); all_pvals.append(m.pvalues[x_var])
            except Exception:
                continue
    if not all_coefs:
        return _empty()
    ests=np.array(all_coefs); pvs=np.array(all_pvals); n=len(ests)
    med=float(np.nanmedian(ests)); confirm=float(np.mean((ests>0)&(pvs<0.05))*100)
    log.info(f"  Fallback spec curve: {n} specs | median={med:.4f} | confirming={confirm:.1f}%")
    return {"n_specs":n,"median_coef":med,"pct_positive":float(np.mean(ests>0)*100),
            "pct_significant":float(np.mean(pvs<0.05)*100),"pct_confirming":confirm,
            "estimates":ests,"p_values":pvs,"figure_path":_plot(ests,pvs,n,med,confirm)}


def _plot(ests, pvs, n, med, confirm):
    idx=np.argsort(ests); es=ests[idx]; ps=pvs[idx]
    colors=["#2ea043" if (e>0 and p<0.05) else "#58a6ff" if e>0 else "#f85149" for e,p in zip(es,ps)]
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,8)); fig.patch.set_facecolor("#0d1117")
    for ax in (ax1,ax2):
        ax.set_facecolor("#0d1117"); ax.spines[:].set_color("#30363d"); ax.tick_params(colors="#8b949e")
    ax1.scatter(range(len(es)),es,c=colors,s=3,alpha=0.6)
    ax1.axhline(0,color="#f85149",lw=1.5,ls="--",alpha=0.8)
    ax1.axhline(med,color="#f8c27d",lw=2,label=f"Median: {med:.4f}")
    ax1.set_title(f"Specification Curve | {n:,} specs | {confirm:.1f}% confirm LSR",color="#c9d1d9",fontweight="bold")
    ax1.set_ylabel("LSR Effect Coefficient",color="#8b949e"); ax1.legend(framealpha=0.3,labelcolor="#c9d1d9")
    ax2.scatter(range(len(ps)),ps,c=colors,s=3,alpha=0.6)
    ax2.axhline(0.05,color="#f85149",lw=1.5,ls="--",label="p=0.05")
    ax2.set_xlabel("Specification (sorted by coefficient)",color="#8b949e"); ax2.set_ylabel("p-value",color="#8b949e")
    ax2.legend(framealpha=0.3,labelcolor="#c9d1d9")
    os.makedirs("outputs",exist_ok=True); path="outputs/spec_curve.png"
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight",facecolor="#0d1117"); plt.close()
    return path


def _empty():
    return {"n_specs":0,"median_coef":np.nan,"pct_positive":np.nan,"pct_significant":np.nan,
            "pct_confirming":np.nan,"estimates":np.array([]),"p_values":np.array([]),"figure_path":None}
