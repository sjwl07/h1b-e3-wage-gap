# H-1B / H-1B1 / E-3 LCA Petitions: Data Science Project

## Data source
- **Dataset:** LCA Programs (H-1B, H-1B1, E-3) Visa Petitions, Kaggle: https://www.kaggle.com/datasets/jishnukoliyadan/lca-programs-h1b-h1b1-e3-visa-petitions (CC BY-NC-SA 4.0)
- **Original source:** US Department of Labor, Office of Foreign Labor Certification, LCA disclosure data: https://www.dol.gov/agencies/eta/foreign-labor/performance
- **Coverage:** Labor Condition Applications, fiscal years 2017–2022
- **Files:** `archive/LCA_FY_2017.csv` … `LCA_FY_2022.csv`, 3,973,349 rows in total (11 columns; 12 from FY2020, which adds `Quarter`), downloaded 2026-10-21

## Setup
The data isn't in this repo (it's 680 MB, and the license asks for it to come from the source). Download it from the Kaggle link above and unzip it so the six files are at `archive/LCA_FY_2017.csv` … `archive/LCA_FY_2022.csv`. Then run each step in the pipeline below in order; together they rebuild `data/`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt    # exact versions: requirements.lock.txt
```

## Pipeline
Run from `h1-b/`:

| Step | Command | Output |
|---|---|---|
| 1. Load and harmonize the 6 years | `.venv/bin/python src/load.py` | `data/raw/lca_all.parquet`, `report/load_log.json`, `report/fy2019_wage_check.md` |
| 2. Split (before any EDA) | `.venv/bin/python src/split.py` | `data/raw/explore.parquet`, `data/raw/confirm.parquet`, `report/split_log.json` |
| 3. Data dictionary | `.venv/bin/python src/data_dictionary.py` | `report/data_dictionary.md` |
| 4. Cleaning | `.venv/bin/python src/clean.py` | `data/processed/explore_clean.parquet`, `data/processed/clean_params.json`, `report/cleaning_log.md` |
| 5. EDA | `.venv/bin/python src/eda.py` | `figures/fig01–fig12.png`, `report/eda.md` |
| 6. Problem statements & pre-registration | `.venv/bin/python src/power.py` | `report/problem_statements.md`, `report/preregistration.md`, `report/power.md` (tests defined in `src/tests_spec.py`, model in `src/fe.py`) |
| 7. Hypothesis tests (confirmation set) | `.venv/bin/python src/inference.py` | `report/inference_results.{md,json}`, `report/cleaning_log_confirm.md`, `data/processed/confirm_clean.parquet`, `figures/fig13–fig14.png` |

**What `load.py` harmonizes:**
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
Figures are read from `figures/`. Names, roll numbers, emails, group name and the GitHub link are red placeholders.
