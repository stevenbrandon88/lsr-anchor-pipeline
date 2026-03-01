"""
tests/test_classifier.py
========================
Unit tests for the LSR classification schema.

Run with:  python -m pytest tests/ -v
Or:        python tests/test_classifier.py   (no pytest needed)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import unittest

from lsr_classifier import apply_lsr_classification, SYMBIOTIC_THRESHOLD, SUCCESS_THRESHOLD


def _make_project(qae="Satisfactory", qos="Satisfactory",
                  bp="Satisfactory", me="Substantial",
                  outcome="Satisfactory", region="Sub-Saharan Africa",
                  country="Kenya", sector="Education",
                  amount=10e6, date="2012-06-01", n=1):
    return pd.DataFrame({
        "quality_at_entry":      [qae] * n,
        "quality_of_supervision":[qos] * n,
        "borrower_performance":  [bp]  * n,
        "me_quality":            [me]  * n,
        "outcome":               [outcome] * n,
        "regionname":            [region] * n,
        "countryname":           [country] * n,
        "sector1":               [sector] * n,
        "totalcommamt":          [amount] * n,
        "boardapprovaldate":     [date] * n,
    })


class TestLSRClassifier(unittest.TestCase):

    # ── Symbiotic classification ───────────────────────────────────────────────
    def test_highly_satisfactory_is_symbiotic(self):
        df = _make_project(qae="Highly Satisfactory", qos="Highly Satisfactory",
                           bp="Highly Satisfactory")
        result = apply_lsr_classification(df)
        self.assertEqual(result["lsr_class"].iloc[0], "symbiotic")
        self.assertEqual(result["T"].iloc[0], 1)

    def test_satisfactory_is_symbiotic(self):
        df = _make_project(qae="Satisfactory", qos="Satisfactory", bp="Satisfactory")
        result = apply_lsr_classification(df)
        self.assertEqual(result["lsr_class"].iloc[0], "symbiotic")

    def test_moderately_satisfactory_is_symbiotic(self):
        """Moderately Satisfactory = 4/6 = at threshold → symbiotic."""
        df = _make_project(qae="Moderately Satisfactory", qos="Moderately Satisfactory",
                           bp="Moderately Satisfactory")
        result = apply_lsr_classification(df)
        self.assertEqual(result["lsr_class"].iloc[0], "symbiotic")

    # ── Extractive classification ─────────────────────────────────────────────
    def test_unsatisfactory_is_extractive(self):
        df = _make_project(qae="Unsatisfactory", qos="Unsatisfactory",
                           bp="Unsatisfactory")
        result = apply_lsr_classification(df)
        self.assertEqual(result["lsr_class"].iloc[0], "extractive")
        self.assertEqual(result["T"].iloc[0], 0)

    def test_highly_unsatisfactory_is_extractive(self):
        df = _make_project(qae="Highly Unsatisfactory", qos="Highly Unsatisfactory",
                           bp="Highly Unsatisfactory")
        result = apply_lsr_classification(df)
        self.assertEqual(result["lsr_class"].iloc[0], "extractive")

    def test_moderately_unsatisfactory_is_extractive(self):
        """Moderately Unsatisfactory = 3/6 = below threshold → extractive."""
        df = _make_project(qae="Moderately Unsatisfactory",
                           qos="Moderately Unsatisfactory",
                           bp="Moderately Unsatisfactory")
        result = apply_lsr_classification(df)
        self.assertEqual(result["lsr_class"].iloc[0], "extractive")

    # ── Outcome / success ─────────────────────────────────────────────────────
    def test_satisfactory_outcome_is_success(self):
        df = _make_project(outcome="Satisfactory")
        result = apply_lsr_classification(df)
        self.assertEqual(result["success"].iloc[0], 1)

    def test_unsatisfactory_outcome_is_failure(self):
        df = _make_project(outcome="Unsatisfactory")
        result = apply_lsr_classification(df)
        self.assertEqual(result["success"].iloc[0], 0)

    def test_moderately_satisfactory_outcome_is_success(self):
        df = _make_project(outcome="Moderately Satisfactory")
        result = apply_lsr_classification(df)
        self.assertEqual(result["success"].iloc[0], 1)

    def test_moderately_unsatisfactory_outcome_is_failure(self):
        df = _make_project(outcome="Moderately Unsatisfactory")
        result = apply_lsr_classification(df)
        self.assertEqual(result["success"].iloc[0], 0)

    # ── LSR_Φ composite score ──────────────────────────────────────────────────
    def test_phi_range(self):
        """LSR_Φ must stay within [1, 6] for all valid ratings."""
        df = pd.concat([
            _make_project(qae="Highly Satisfactory", qos="Highly Satisfactory",
                         bp="Highly Satisfactory", me="High"),
            _make_project(qae="Highly Unsatisfactory", qos="Highly Unsatisfactory",
                         bp="Highly Unsatisfactory", me="Negligible"),
        ], ignore_index=True)
        result = apply_lsr_classification(df)
        self.assertTrue((result["lsr_phi"].dropna() >= 1).all())
        self.assertTrue((result["lsr_phi"].dropna() <= 6).all())

    def test_phi_monotonic(self):
        """Higher ratings → higher LSR_Φ."""
        high = apply_lsr_classification(_make_project(
            qae="Highly Satisfactory", qos="Highly Satisfactory",
            bp="Highly Satisfactory", me="High"))
        low = apply_lsr_classification(_make_project(
            qae="Highly Unsatisfactory", qos="Highly Unsatisfactory",
            bp="Highly Unsatisfactory", me="Negligible"))
        self.assertGreater(high["lsr_phi"].iloc[0], low["lsr_phi"].iloc[0])

    def test_weights_sum_to_one(self):
        from lsr_classifier import WEIGHTS
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0, places=5)

    # ── Bulk / population tests ────────────────────────────────────────────────
    def test_bulk_classification(self):
        """200 symbiotic + 100 extractive should classify cleanly."""
        df = pd.concat([
            _make_project(qae="Satisfactory", qos="Satisfactory",
                         bp="Satisfactory", n=200),
            _make_project(qae="Unsatisfactory", qos="Unsatisfactory",
                         bp="Unsatisfactory", n=100),
        ], ignore_index=True)
        result = apply_lsr_classification(df)
        n_sym = (result["lsr_class"] == "symbiotic").sum()
        n_ext = (result["lsr_class"] == "extractive").sum()
        self.assertEqual(n_sym, 200)
        self.assertEqual(n_ext, 100)

    def test_known_lsr_direction(self):
        """Symbiotic projects must have higher success rate than extractive."""
        df = pd.concat([
            _make_project(qae="Highly Satisfactory", qos="Satisfactory",
                         bp="Satisfactory", outcome="Satisfactory", n=100),
            _make_project(qae="Highly Unsatisfactory", qos="Unsatisfactory",
                         bp="Unsatisfactory", outcome="Unsatisfactory", n=100),
        ], ignore_index=True)
        result = apply_lsr_classification(df)
        sym_rate = result.loc[result["lsr_class"] == "symbiotic", "success"].mean()
        ext_rate = result.loc[result["lsr_class"] == "extractive", "success"].mean()
        self.assertGreater(sym_rate, ext_rate,
            f"Expected sym_rate > ext_rate, got {sym_rate:.2f} vs {ext_rate:.2f}")

    # ── Missing data handling ──────────────────────────────────────────────────
    def test_missing_me_quality_handled(self):
        """Pipeline should not crash if M&E quality is missing."""
        df = _make_project(qae="Satisfactory", qos="Satisfactory", bp="Satisfactory")
        df["me_quality"] = np.nan
        result = apply_lsr_classification(df)
        self.assertIn(result["lsr_class"].iloc[0], ["symbiotic", "extractive", None])

    def test_not_rated_produces_nan(self):
        """'Not Rated' should produce NaN phi and no classification."""
        df = _make_project(qae="Not Rated", qos="Not Rated", bp="Not Rated")
        result = apply_lsr_classification(df)
        # With all NaN ratings, phi should be NaN
        self.assertTrue(result["lsr_phi"].isna().all() or
                       result["lsr_class"].iloc[0] is None or
                       pd.isna(result["lsr_class"].iloc[0]))

    # ── Column output completeness ─────────────────────────────────────────────
    def test_required_output_columns_present(self):
        df = _make_project()
        result = apply_lsr_classification(df)
        for col in ["lsr_phi", "lsr_class", "T", "success"]:
            self.assertIn(col, result.columns, f"Missing output column: {col}")


class TestColumnMapping(unittest.TestCase):
    """Test that wb_data_fetcher correctly maps actual IEG CSV column names."""

    def test_ieg_short_columns_mapped(self):
        """QaE, QoS, Borr Perf, M&E Qual must map to internal names."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from wb_data_fetcher import _standardise_columns

        # Simulate what the actual WB IEG CSV looks like
        df_ieg = pd.DataFrame({
            "Proj ID":   ["P001"],
            "Proj Name": ["Test Project"],
            "Country":   ["Kenya"],
            "Region":    ["Sub-Saharan Africa"],
            "Approval FY": [2015],
            "QaE":         ["Satisfactory"],        # ← actual IEG column name
            "QoS":         ["Satisfactory"],        # ← actual IEG column name
            "Borr Perf":   ["Satisfactory"],        # ← actual IEG column name
            "M&E Qual":    ["Substantial"],         # ← actual IEG column name
            "Bank Perf":   ["Satisfactory"],
            "Outcome":     ["Satisfactory"],
            "Commit Amt (USD)": [10000000],
        })

        result = _standardise_columns(df_ieg)

        self.assertIn("quality_at_entry", result.columns,
                      "QaE not mapped to quality_at_entry")
        self.assertIn("quality_of_supervision", result.columns,
                      "QoS not mapped to quality_of_supervision")
        self.assertIn("borrower_performance", result.columns,
                      "Borr Perf not mapped to borrower_performance")
        self.assertIn("me_quality", result.columns,
                      "M&E Qual not mapped to me_quality")

        # Values must survive the mapping (not become NaN)
        self.assertEqual(result["quality_at_entry"].iloc[0], "Satisfactory")
        self.assertEqual(result["borrower_performance"].iloc[0], "Satisfactory")

    def test_full_pipeline_with_ieg_format(self):
        """End-to-end: IEG-format DataFrame → classified projects."""
        from wb_data_fetcher import _standardise_columns
        from lsr_classifier import apply_lsr_classification

        df_ieg = pd.DataFrame({
            "Proj ID": ["P001", "P002", "P003"],
            "Country": ["Kenya", "India", "Ghana"],
            "Region":  ["Sub-Saharan Africa", "South Asia", "Sub-Saharan Africa"],
            "Approval FY": [2010, 2012, 2015],
            "QaE":       ["Satisfactory", "Satisfactory", "Unsatisfactory"],
            "QoS":       ["Satisfactory", "Satisfactory", "Unsatisfactory"],
            "Borr Perf": ["Satisfactory", "Satisfactory", "Unsatisfactory"],
            "M&E Qual":  ["Substantial", "Substantial", "Modest"],
            "Outcome":   ["Satisfactory", "Satisfactory", "Unsatisfactory"],
            "Sector (Code)": ["Education", "Health", "Energy"],
            "Commit Amt (USD)": [5e6, 10e6, 8e6],
        })

        df_std = _standardise_columns(df_ieg)
        df_cls = apply_lsr_classification(df_std)

        classified = df_cls["lsr_class"].notna().sum()
        self.assertGreater(classified, 0,
            "Pipeline produced 0 classified projects with valid IEG data. "
            "Column mapping is broken.")
        # Note: "Proj ID" gets renamed to "project_id" by _standardise_columns
        self.assertEqual(df_cls.loc[df_cls["project_id"] == "P003", "lsr_class"].iloc[0],
                        "extractive", "Unsatisfactory project should be extractive")


if __name__ == "__main__":
    print("Running LSR classification tests...")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestLSRClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestColumnMapping))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print()
    if result.wasSuccessful():
        print("✓ All tests passed")
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")
    sys.exit(0 if result.wasSuccessful() else 1)
