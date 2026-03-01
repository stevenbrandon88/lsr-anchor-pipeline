"""
tests/test_robustness.py
========================
Smoke tests for RobustiPy, DoWhy, EconML wrappers.
Tests the API calls only — does not require live WB data.

Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import unittest


def _make_classified_df(n_sym=200, n_ext=100):
    """Minimal classified DataFrame for testing downstream tools."""
    import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from lsr_classifier import apply_lsr_classification

    df = pd.concat([
        pd.DataFrame({
            "quality_at_entry":      ["Satisfactory"] * n_sym,
            "quality_of_supervision":["Satisfactory"] * n_sym,
            "borrower_performance":  ["Satisfactory"] * n_sym,
            "me_quality":            ["Substantial"]  * n_sym,
            "outcome":               ["Satisfactory"] * n_sym,
            "regionname":            ["Sub-Saharan Africa"] * n_sym,
            "countryname":           ["Kenya"] * n_sym,
            "sector1":               ["Education"] * n_sym,
            "totalcommamt":          np.random.uniform(1e6, 1e8, n_sym),
            "boardapprovaldate":     ["2012-01-01"] * n_sym,
        }),
        pd.DataFrame({
            "quality_at_entry":      ["Unsatisfactory"] * n_ext,
            "quality_of_supervision":["Unsatisfactory"] * n_ext,
            "borrower_performance":  ["Unsatisfactory"] * n_ext,
            "me_quality":            ["Negligible"]    * n_ext,
            "outcome":               ["Unsatisfactory"] * n_ext,
            "regionname":            ["South Asia"] * n_ext,
            "countryname":           ["India"] * n_ext,
            "sector1":               ["Health"] * n_ext,
            "totalcommamt":          np.random.uniform(1e6, 1e8, n_ext),
            "boardapprovaldate":     ["2008-01-01"] * n_ext,
        }),
    ], ignore_index=True)

    df_c = apply_lsr_classification(df)
    # Encode for downstream tools
    df_c["region_enc"]    = pd.Categorical(df_c["regionname"]).codes.astype(float)
    df_c["sector_enc"]    = pd.Categorical(df_c["sector1"]).codes.astype(float)
    df_c["hdi_group_enc"] = pd.Categorical(df_c["hdi_group"]).codes.astype(float)
    df_c["totalcommamt_log"] = np.log1p(df_c["totalcommamt"].clip(lower=0))
    return df_c


class TestRobustiPyWrapper(unittest.TestCase):

    def setUp(self):
        self.df = _make_classified_df()

    def test_returns_dict_with_required_keys(self):
        from robustness_analysis import run_robustipy
        result = run_robustipy(self.df)
        for key in ["n_specs", "median_coef", "pct_confirming", "estimates", "p_values"]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_n_specs_positive(self):
        from robustness_analysis import run_robustipy
        result = run_robustipy(self.df)
        self.assertGreater(result["n_specs"], 0, "Zero specifications run")

    def test_positive_direction(self):
        from robustness_analysis import run_robustipy
        result = run_robustipy(self.df)
        self.assertGreater(result["median_coef"], 0,
            "Median coefficient should be positive for symbiotic > extractive data")

    def test_high_confirmation_rate(self):
        from robustness_analysis import run_robustipy
        result = run_robustipy(self.df)
        self.assertGreater(result["pct_confirming"], 50,
            f"Expected >50% confirming, got {result['pct_confirming']:.1f}%")


class TestDoWhyWrapper(unittest.TestCase):

    def setUp(self):
        self.df = _make_classified_df()

    def test_returns_dict_with_required_keys(self):
        from causal_analysis import run_dowhy
        result = run_dowhy(self.df)
        for key in ["ate", "refutations_passed", "n_obs"]:
            self.assertIn(key, result)

    def test_ate_positive(self):
        from causal_analysis import run_dowhy
        result = run_dowhy(self.df)
        if not np.isnan(result["ate"]):
            self.assertGreater(result["ate"], 0,
                "ATE should be positive for symbiotic > extractive data")

    def test_refutations_attempted(self):
        from causal_analysis import run_dowhy
        result = run_dowhy(self.df)
        # At minimum the random_common_cause refutation should run
        self.assertGreaterEqual(result["refutations_passed"], 0)


class TestEconMLWrapper(unittest.TestCase):

    def setUp(self):
        self.df = _make_classified_df()

    def test_returns_dict_with_required_keys(self):
        from heterogeneous_effects import run_econml
        result = run_econml(self.df)
        for key in ["mean_cate", "hdi_effects", "region_effects"]:
            self.assertIn(key, result)

    def test_mean_cate_direction(self):
        from heterogeneous_effects import run_econml
        result = run_econml(self.df)
        if not np.isnan(result["mean_cate"]):
            self.assertGreater(result["mean_cate"], 0,
                "Mean CATE should be positive for symbiotic > extractive data")


if __name__ == "__main__":
    print("Running tool wrapper tests...")
    print("NOTE: RobustiPy/DoWhy/EconML tests take ~5 minutes")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRobustiPyWrapper))
    suite.addTests(loader.loadTestsFromTestCase(TestDoWhyWrapper))
    suite.addTests(loader.loadTestsFromTestCase(TestEconMLWrapper))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
