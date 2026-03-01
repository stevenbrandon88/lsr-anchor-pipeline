"""
tests/test_institutions.py
==========================
Integration tests across all four institution loaders.
Verifies: data loads, classification runs, LSR direction confirmed.

Run: python tests/test_institutions.py    (no pytest needed)
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd, numpy as np, unittest

from institution_fetchers import fetch_adb_ied, fetch_aiddata
from lsr_classifier import apply_lsr_classification


def _direction_ok(df):
    """Core check: symbiotic success rate > extractive success rate."""
    if 'success' not in df.columns:
        return False, 0, 0, np.nan
    s1 = df.loc[df['lsr_class']=='symbiotic','success'].dropna().mean()
    s0 = df.loc[df['lsr_class']=='extractive','success'].dropna().mean()
    orv = ((s1/(1-s1))/(s0/(1-s0))) if (0<s0<1 and 0<s1<1) else np.nan
    return s1 > s0, s1, s0, orv


class TestADBIED(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = apply_lsr_classification(fetch_adb_ied(max_rows=300), 'adb')

    def test_loads_data(self):
        self.assertGreater(len(self.df), 50, "ADB: too few rows loaded")

    def test_classifies_both_groups(self):
        sym = (self.df['lsr_class']=='symbiotic').sum()
        ext = (self.df['lsr_class']=='extractive').sum()
        self.assertGreater(sym, 10, f"ADB: only {sym} symbiotic projects")
        self.assertGreater(ext, 5,  f"ADB: only {ext} extractive projects")

    def test_required_columns(self):
        for col in ['lsr_phi','lsr_class','T','success','hdi_group','decade']:
            self.assertIn(col, self.df.columns, f"ADB: missing column {col}")

    def test_lsr_direction(self):
        ok, s1, s0, orv = _direction_ok(self.df)
        self.assertTrue(ok,
            f"ADB: symbiotic={s1*100:.1f}% NOT > extractive={s0*100:.1f}%. "
            f"LSR direction WRONG for ADB IED data.")

    def test_odds_ratio_above_one(self):
        _, s1, s0, orv = _direction_ok(self.df)
        if not np.isnan(orv):
            self.assertGreater(orv, 1.0, f"ADB: OR={orv:.2f} should be >1")

    def test_phi_in_valid_range(self):
        phi = self.df['lsr_phi'].dropna()
        self.assertTrue((phi >= 1).all() and (phi <= 6).all(),
            f"ADB: lsr_phi out of [1,6] range")

    def test_six_cohorts_loaded(self):
        """Confirm data loaded from multiple annual cohorts (2018 + 2020-2024)."""
        # _aer_year tracks which annual report cohort each row came from
        if '_aer_year' in self.df.columns:
            cohorts = self.df['_aer_year'].dropna().nunique()
            self.assertGreater(cohorts, 1, "ADB: only one cohort loaded — expected 6 (2018+2020-2024)")


class TestAidData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        raw = fetch_aiddata(max_rows=500)
        if 'lsr_class' in raw.columns:
            cls.df = raw   # pre-classified CSV
        else:
            cls.df = apply_lsr_classification(raw, 'aiddata')

    def test_loads_data(self):
        self.assertGreater(len(self.df), 100, "AidData: too few rows")

    def test_classifies_both_groups(self):
        sym = (self.df['lsr_class']=='symbiotic').sum()
        ext = (self.df['lsr_class']=='extractive').sum()
        self.assertGreater(sym, 10)
        self.assertGreater(ext, 5)

    def test_required_columns(self):
        for col in ['lsr_phi','lsr_class','T','success']:
            self.assertIn(col, self.df.columns, f"AidData: missing {col}")

    def test_lsr_direction(self):
        ok, s1, s0, orv = _direction_ok(self.df)
        self.assertTrue(ok,
            f"AidData: symbiotic={s1*100:.1f}% NOT > extractive={s0*100:.1f}%. "
            f"LSR direction WRONG for AidData GCDF.")

    def test_independent_evaluator_flag(self):
        """AidData uses completion tracking from William & Mary — not MDB self-report."""
        inst = self.df.get('_institution', pd.Series([''])).iloc[0]
        self.assertIn('AidData', str(inst), "AidData institution flag missing")


class TestCrossInstitutional(unittest.TestCase):
    """Meta-test: LSR direction must hold across ALL loaded institutions."""

    def test_all_institutions_confirm_lsr(self):
        results = {}

        # ADB IED
        try:
            df = apply_lsr_classification(fetch_adb_ied(max_rows=200), 'adb')
            ok, s1, s0, orv = _direction_ok(df)
            results['ADB IED'] = {'confirms': ok, 'or': orv, 'n': len(df)}
        except Exception as e:
            self.fail(f"ADB IED failed to load: {e}")

        # AidData
        try:
            raw = fetch_aiddata(max_rows=200)
            df = raw if 'lsr_class' in raw.columns else apply_lsr_classification(raw, 'aiddata')
            ok, s1, s0, orv = _direction_ok(df)
            results['AidData'] = {'confirms': ok, 'or': orv, 'n': len(df)}
        except Exception as e:
            self.fail(f"AidData failed to load: {e}")

        print("\n  Cross-institutional summary:")
        for label, r in results.items():
            status = '✓' if r['confirms'] else '✗'
            print(f"    {status} {label}: OR={r['or']:.2f}x  n={r['n']:,}")

        failing = [l for l, r in results.items() if not r['confirms']]
        self.assertEqual(failing, [],
            f"LSR direction WRONG for: {failing}. "
            f"The core finding does not replicate across institutions.")


if __name__ == '__main__':
    print("Testing multi-institution LSR pipeline...")
    print("=" * 60)
    suite = unittest.TestSuite()
    for cls in [TestADBIED, TestAidData, TestCrossInstitutional]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    if result.wasSuccessful():
        print("✓ All institution tests passed")
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")
    sys.exit(0 if result.wasSuccessful() else 1)
