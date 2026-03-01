# LSR Anchor Pipeline

**Law of Symbiotic Resilience — Multi-Institutional Empirical Validation**  
Author: Steven Brandon | PhD Candidate, Griffith University / QUT  
Supervisors: Prof. Brendan Mackey, A/Prof. Alexandr Akimov, A/Prof. Shyama Ratnasiri  
SSRN: https://ssrn.com/author=7801909

---

## What This Does

Tests one question empirically: *Does symbiotic project design cause better development outcomes?*

Across **three independent institutions** and **four independent datasets**, using three independent analytical tools from Microsoft Research and academic literature.

| Institution | Evaluator | N Projects | OR | ATE | Spec Curve | Refutations |
|---|---|---|---|---|---|---|
| World Bank IEG | WB Independent Evaluation Group | ~10,000 | TBD | TBD | TBD | TBD |
| ADB IED | ADB Independent Evaluation Dept (2020–2024) | 2,388 | 2.53× | +21.0pp | 3,200 specs / 100% | 3/3 ✓ |
| AidData GCDF | William & Mary (Chinese DFI tracker) | 8,923 | 8.32× | +1.9pp | 6,400 specs / 100% | 3/3 ✓ |

**Credibility architecture:** The researcher's sole contribution is the classification schema in `lsr_classifier.py`. Data comes from independent institutions. Analysis runs on Microsoft Research tools (DoWhy, EconML) and peer-reviewed statistical methodology (RobustiPy, Simonsohn et al. 2020).

---

## Setup

### Requirements
- Python 3.10, 3.11, or 3.12

### Step 1 — Clone / unzip

```bash
git clone https://github.com/stevenbrandon88/lsr-anchor-pipeline.git
cd lsr-anchor-pipeline
```

### Step 2 — Virtual environment

```bash
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

### Step 3 — Install

```bash
pip install -r requirements.txt
```

### Step 4 — Add data files

**World Bank IEG** (auto-fetched on first run, or download manually):
- Download: https://ieg.worldbankgroup.org/evaluations/data → "Project Performance Ratings"
- Save as: `data/wb/ieg_ratings.csv`

**ADB IED** (download free from ADB, place in `data/adb/`):
- https://www.adb.org/what-we-do/evaluation/data
- Files needed (any or all):
  - `AER-2020-Success-Rates.xlsx`
  - `aer-2023_success-rate-database.xlsx`
  - `aer-2024-success-rates-database.xlsx`

**AidData GCDF v2.0** (place in `data/aiddata/`):
- https://www.aiddata.org/data/aiddatas-global-chinese-development-finance-dataset-version-2-0
- File: `AidDatasGlobalChineseDevelopmentFinanceDataset_v2_0.xlsx`

---

## Running

```bash
# All available institutions (~20-30 min)
python run_pipeline.py

# Specific institutions only
python run_pipeline.py --institutions adb aid

# Fast test with 200 projects per institution (~3 min)
python run_pipeline.py --max 200

# Use make
make test        # unit tests (10 seconds)
make run-test    # 200 projects per institution
make run         # full run
```

---

## Outputs

| File | Contents |
|---|---|
| `outputs/lsr_results_[timestamp].xlsx` | Full results: summary, per-institution tables, cross-institutional comparison |
| `outputs/cross_institutional.png` | 3-panel comparison: OR / ATE / spec curve confirmation across institutions |
| `outputs/spec_curve.png` | Specification curve (sorted coefficients across all control combinations) |
| `outputs/hte_effects.png` | Heterogeneous treatment effects by HDI group and region |
| `logs/pipeline_[timestamp].log` | Full audit trail with timestamps |

---

## Repository Structure

```
lsr-anchor-pipeline/
├── run_pipeline.py            ← Entry point
├── lsr_classifier.py          ← LSR schema (the sole contribution)
├── institution_fetchers.py    ← Data loaders: WB IEG, ADB IED, AidData
├── cross_institutional.py     ← Cross-institution comparison
├── robustness_analysis.py     ← RobustiPy specification curve
├── causal_analysis.py         ← DoWhy causal identification
├── heterogeneous_effects.py   ← EconML heterogeneous effects
├── output_generator.py        ← Excel + figure generation
├── requirements.txt
├── Makefile
├── tests/
│   ├── test_classifier.py     ← 20 unit tests (run without data)
│   └── test_robustness.py     ← Tool wrapper smoke tests
└── data/
    ├── wb/                    ← ieg_ratings.csv (auto-populated)
    ├── adb/                   ← AER xlsx files (place here)
    └── aiddata/               ← GCDF xlsx/csv (place here)
```

---

## For Supervisors / Peer Reviewers

One-command reproduction:

```bash
git clone https://github.com/stevenbrandon88/lsr-anchor-pipeline.git
cd lsr-anchor-pipeline
pip install -r requirements.txt
make run
```

**Data provenance:**  
- World Bank IEG: fetched live from `ieg.worldbankgroup.org`. Timestamp logged.  
- ADB IED: published annual evaluation reports, 2020–2024.  
- AidData GCDF: published dataset, William & Mary AidData lab v2.0 (2000–2017).

**Methodological provenance:**  
- Specification robustness: RobustiPy — Simonsohn, Simmons & Nelson (2020) *Nature Human Behaviour*  
- Causal identification: DoWhy — Sharma & Kiciman (2020), Microsoft Research  
- Heterogeneous effects: EconML — Battocchi et al. (2019), Microsoft Research  

**Researcher's sole contribution:** `lsr_classifier.py`

---

## Citation

Brandon, S. (2025). LSR Anchor Pipeline: Multi-institutional empirical validation of the Law of Symbiotic Resilience. SSRN 7801909.
