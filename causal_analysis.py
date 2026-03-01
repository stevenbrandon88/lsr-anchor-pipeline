"""
causal_analysis.py - DoWhy causal identification wrapper.
API: backdoor.propensity_score_stratification
"""
import pandas as pd, numpy as np, logging
log = logging.getLogger(__name__)
CONFOUNDERS = ["decade","region_enc","sector_enc","totalcommamt_log","is_SIDS","is_FCAS","is_fcs"]

def run_dowhy(df: pd.DataFrame) -> dict:
    try:
        from dowhy import CausalModel
    except ImportError:
        log.error("DoWhy not installed: pip install dowhy"); return _empty()
    available = [c for c in CONFOUNDERS if c in df.columns]
    df_c = df[["success","T"]+available].dropna().copy()
    if len(df_c) < 100:
        log.warning(f"  DoWhy: only {len(df_c)} rows — skipping"); return _empty()
    log.info(f"  DoWhy: n={len(df_c):,} | confounders={available}")
    nodes = ["T","success"]+available
    gml = "graph [directed 1\n"
    for i,n in enumerate(nodes): gml += f'  node [id {i} label "{n}"]\n'
    ti=nodes.index("T"); yi=nodes.index("success")
    for c in available:
        ci=nodes.index(c); gml+=f'  edge [source {ci} target {ti}]\n  edge [source {ci} target {yi}]\n'
    gml += f'  edge [source {ti} target {yi}]\n]'
    try:
        model = CausalModel(data=df_c, treatment="T", outcome="success", graph=gml)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(estimand,
            method_name="backdoor.propensity_score_stratification",
            method_params={"num_strata":10})
        ate = float(estimate.value)
        log.info(f"  DoWhy ATE: {ate:.4f} ({ate*100:.1f}pp)")
        passed = 0; refutation_results = {}
        for method, params, is_pass in [
            ("random_common_cause", {}, lambda new,ate: abs(new-ate)<abs(ate)*0.3),
            ("placebo_treatment_refuter", {"placebo_type":"permute","num_simulations":5},
             lambda new,ate: abs(new)<abs(ate)*0.5),
            ("data_subset_refuter", {"subset_fraction":0.8,"num_simulations":5},
             lambda new,ate: abs(new-ate)<abs(ate)*0.3),
        ]:
            try:
                ref = model.refute_estimate(estimand, estimate, method_name=method, **params)
                result = is_pass(ref.new_effect, ate)
                if result: passed += 1
                refutation_results[method] = {"new_effect":float(ref.new_effect),"passed":result}
                log.info(f"  {method}: {'PASS' if result else 'FAIL'} (new={ref.new_effect:.4f})")
            except Exception as e:
                log.warning(f"  Refutation {method} failed: {e}")
                refutation_results[method] = {"new_effect":np.nan,"passed":False}
        log.info(f"  Refutations: {passed}/3")
        return {"ate":ate,"refutations_passed":passed,"refutation_results":refutation_results,"n_obs":len(df_c),"confounders_used":available}
    except Exception as e:
        log.error(f"  DoWhy error: {e}"); return _empty()

def _empty():
    return {"ate":np.nan,"refutations_passed":0,"refutation_results":{},"n_obs":0,"confounders_used":[]}
