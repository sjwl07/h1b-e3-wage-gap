# Same job, higher wage? E-3 versus H-1B filings (FY2017–2022)

CSE558 Data Science, IIIT-Delhi, Monsoon 2026 · Group **Ramanujan et al.** (Shivam Jhunjhunuwala, Ramanujan Gopalan, Saanvi Singh, Mohammad Umar, Mayank Kumar)

**Question.** For the same job, state and year, do employers file Australian (E-3) workers at a different prevailing wage than H-1B workers, and are E-3 filings denied more often? Four tests, run on a 20% held-out set (794,670 filings):

| Test | Result | Verdict |
|---|---|---|
| T1: wage, same job × state × year | E-3 **+9.1%** (95% CI 7.7 to 10.5) | meaningful (MID 5%) |
| T2: wage, same employer too | E-3 **+6.8%** (5.3 to 8.3) | meaningful, but concentrated in some employers (median employer +1.5%) |
| T3: denial, same employer size × year | E-3 **+3.79 pp** (3.26 to 4.31) | meaningful (MID 0.5 pp) |
| T4: denial, same employer × year | E-3 **+0.51 pp** (0.22 to 0.81) | possibly too small |

All four reject H₀ after Holm correction. The full write-up is `report/latex/report.pdf`.

## Data source
- **Dataset:** LCA Programs (H-1B, H-1B1, E-3) Visa Petitions, Kaggle: https://www.kaggle.com/datasets/jishnukoliyadan/lca-programs-h1b-h1b1-e3-visa-petitions (CC BY-NC-SA 4.0)
- **Original source:** US Department of Labor, Office of Foreign Labor Certification, LCA disclosure data: https://www.dol.gov/agencies/eta/foreign-labor/performance
- **Coverage:** Labor Condition Applications, fiscal years 2017–2022
- **Files:** `archive/LCA_FY_2017.csv` … `LCA_FY_2022.csv`, 3,973,349 rows in total (11 columns; 12 from FY2020, which adds `Quarter`), downloaded 2026-10-21

## Setup
The data isn't in this repo (it's 680 MB, and the license asks for it to come from the source). Download it from the Kaggle link above and unzip it so the six files are at `archive/LCA_FY_2017.csv` … `archive/LCA_FY_2022.csv`. Then run each step in the pipeline below in order; together they rebuild `data/`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt    # exact versions used (Python 3.12.3)
```

## Pipeline
Run from the repository root (`h1b-e3-wage-gap/`):

| Step | Command | Output |
|---|---|---|
| 1. Load the 6 years and match their columns | `.venv/bin/python src/load.py` | `data/raw/lca_all.parquet`, `report/load_log.json`, `report/fy2019_wage_check.md` |
| 2. Split (before any EDA) | `.venv/bin/python src/split.py` | `data/raw/explore.parquet`, `data/raw/confirm.parquet`, `report/split_log.json` |
| 3. Data dictionary | `.venv/bin/python src/data_dictionary.py` | `report/data_dictionary.md` |
| 4. Cleaning | `.venv/bin/python src/clean.py` | `data/processed/explore_clean.parquet`, `data/processed/clean_params.json`, `report/cleaning_log.md`, `report/soc_fuzzy_merges.md` |
| 5. EDA | `.venv/bin/python src/eda.py` | `figures/fig01–fig12.png`, `figures/fig15_basic_questions.png`, `report/eda.md` |
| 6. Power check for the planned tests | `.venv/bin/python src/power.py` | `report/power.md` (tests defined in `src/tests_spec.py`, model in `src/fe.py`; the problem statements and test plan are written by hand in `report/problem_statements.md` and `report/preregistration.md`) |
| 7. Hypothesis tests (confirmation set) | `.venv/bin/python src/inference.py` | `report/inference_results.{md,json}`, `report/cleaning_log_confirm.md`, `data/processed/confirm_clean.parquet`, `figures/fig13–fig14.png` |

**What `load.py` fixes so the six years match:**
- Case-status labels are mapped to 4 levels.
- `Worksite` and `Employer_Location` are split into city and state.
- `employer_key` is the upper-cased name with punctuation and legal suffixes removed, so `AMAZON.COM SERVICES, INC.` and `Amazon.com Services Inc` get the same key. This takes 255,125 raw names down to 165,799 keys.
- **FY2019's wage column is the offered wage, not the prevailing wage** (see `report/fy2019_wage_check.md`). It's stored as `wage_offered_fy2019`, and `prevailing_wage` is missing for FY2019.

**Split:**
- Seed `42` (`src/config.py`).
- 80/20 `StratifiedGroupKFold`, grouped by `employer_key` (no employer is in both sets) and stratified by fiscal year × case status.
- Result: 3,178,679 exploration rows and 794,670 confirmation rows.

**`confirm.parquet` is held out:** don't load it for EDA or cleaning decisions. It's used only for confirming the inference results and for final model evaluation.

## Midterm report
`report/latex/report.tex` (built on the course template) compiles to `report/latex/report.pdf`:
```bash
cd report/latex && pdflatex report.tex && pdflatex report.tex
```
Figures are read from `figures/`.
