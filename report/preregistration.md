# Hypothesis test plan

The tests were fixed on 2026-10-21, **before** the confirmation set was cleaned or opened. This file is the current version of that plan; the original wording is in git history (first committed in `2247107`). Since then the reported checks were simplified (see the end), but no hypothesis, α, MID or decision rule changed.

The hypotheses came from EDA on the exploration set, so testing them on that same data would be circular. All four tests run once, on the held-out confirmation set (`data/raw/confirm.parquet`, 794,670 filings from 33,314 employers that never appear in the exploration set). It's cleaned with the rules fitted on the exploration set (`data/processed/clean_params.json`); nothing is re-fitted on it. The populations and models are defined in code in `src/tests_spec.py`, which `power.py` and `inference.py` both import.

## Common settings

| Item | Choice | Reason |
|---|---|---|
| Groups | E-3 Australian vs H-1B (variable `e3` = 1 for E-3) | H-1B1 (Chile/Singapore) is 0.4% of filings and a separate program. |
| Model | `y = β·e3 + cell intercept + error`; cell intercepts removed by subtracting cell means | β becomes the E-3 minus H-1B gap **inside** the same cell, so differences between cells (occupation, state, year, employer) can't create a gap. |
| Standard errors | **Clustered by employer** (`employer_key`); t-test with employers − 1 degrees of freedom | One employer's filings are near-copies (Fig. 12: 55% of rows at the largest filers have an identical twin), so they aren't independent; employers are the independent units. The split was grouped by employer for the same reason. |
| Significance | Family-wise α = 0.05, **Holm** correction across the 4 tests | Four tests at 5% each would give up to a 19% chance of at least one false alarm; Holm keeps it at 5% overall. |
| Sidedness | **Two-sided** for all tests | EDA suggested directions (E-3 higher wage, more denials), but choosing one-sided tests after looking at the data would fit the test to the data, and a reversal would be a finding in itself. |
| **MID, wage (T1, T2)** | **5%** in prevailing wage, i.e. \|β\| ≥ log(1.05) = 0.0488 | About $4,400 at the median wage ($88k); about one year of wage growth in this period (+33% over five years); about half of the gap between "senior" and other titles in the same job (Fig. 9). A smaller gap wouldn't put a worker at a different wage level. |
| **MID, denial (T3, T4)** | **0.5 percentage points** in denial probability | The H-1B denial rate is 0.78%, so +0.5 pp means a ≥1.6× higher risk. It's also more than half the entire FY2017–22 decline in the denial rate (1.36% → 0.49%). |
| Estimate | β with 95% CI; wage results also as exp(β) − 1 (% difference) | Reported for every test, whether or not it rejects. |

**Interpreting each result** (95% CI against ±MID):

| 95% CI vs ±MID | Conclusion |
|---|---|
| entirely beyond ±MID | **Meaningful** |
| includes the MID, and H₀ rejected | **Possibly too small** to matter |
| entirely inside ±MID | **Negligible** |
| crosses ±MID, H₀ not rejected | **Inconclusive**: neither shown to matter nor shown negligible |

With ~800k filings, almost any difference will be statistically significant; the comparison of the CI with the MID is what decides whether it matters.

**Power** (`report/power.md`): the chance of detecting a gap exactly the size of the MID, using exploration-set SEs scaled up to the confirmation set's 4× fewer employers, at α = 0.05/4 (the strictest level Holm can apply): T1 1.00, T2 0.999, T3 **0.16**, T4 0.93. T3 can only reliably detect gaps of about 1.1 pp or more, so a non-rejection of T3 would be read as *inconclusive*, never as "no difference".

---

## T1. E-3 vs H-1B prevailing wage, same occupation × state × year (P1)

- **Population:** E-3 and H-1B filings in FY2017–18 and FY2020–22 (FY2019 has no prevailing wage) with a cleaned wage and a known worksite.
- **Outcome:** log annual prevailing wage. **Cells:** SOC title × worksite state × fiscal year.
- **H₀:** β = 0 (within the same SOC × state × year, mean log PW is equal for E-3 and H-1B filings)
- **H₁:** β ≠ 0 (two-sided)
- **Test:** **t-test on β**, employer-clustered SE, df = employers − 1.
- **Why it fits:** the comparison has to hold occupation, place and year fixed (Figs. 2–4), and cell intercepts do that for thousands of cells without assuming anything about how cells differ. Log PW is symmetric (skew −0.04), and with thousands of employers the clustered t is reliable.
- **Other tests considered:**
  - *Welch two-sample t-test on relative PW:* the same comparison, but it treats every filing as independent, which they aren't, so it understates the SE. Run as a check.
  - *Mann–Whitney U:* tests a shift in ranks, which can't be compared with a 5% MID; log wage is already symmetric, so ranks add nothing; also assumes independence.
  - *Raw comparison without cells:* mixes occupation, state and year composition into the gap.
- **Checks:** Welch t; bootstrap p-value; leaving out the largest employer.

## T2. The same comparison within the same employer (P1)

- **Cells:** employer × SOC title × worksite state × fiscal year. Only employers that file both E-3 and H-1B for the same job, state and year contribute (in exploration: 2,828 employers and ~18k E-3 filings).
- **H₀:** β = 0 (the same employer files the same prevailing wage for E-3 and H-1B workers in the same job, state and year)
- **H₁:** β ≠ 0 (two-sided)
- **Test:** as T1, with the finer cells.
- **Why it fits:** it removes everything about the employer (pay policy, industry, lawyer), so any gap left reflects how one employer files for the two groups.
- **Estimand caveat:** it describes employers that hire both groups for the same job, which are larger than average; it isn't the average E-3 filing.
- **Other tests considered:**
  - *Paired t-test on cell means:* weights a cell with 1 E-3 and 500 H-1B filings the same as a balanced one. Run as a check in the form of one gap per employer, so employers are the independent units.
  - *Wilcoxon signed-rank:* the rank version of the paired t; it would repeat that check.
  - *Matching each E-3 filing to an H-1B filing of the same employer and cell:* discards most H-1B filings for no gain in bias.
- **Checks:** paired t on employer gaps; bootstrap p-value; leaving out the largest employer.

## T3. E-3 vs H-1B denial rate, same employer-size group × year (P2)

- **Population:** all E-3 and H-1B filings, FY2017–22 (case status is known for every year).
- **Outcome:** `denied` (1 = DENIED, else 0). **Cells:** employer-size group (filings per year: 1, 2–10, 11–100, 101–1000, > 1000) × fiscal year.
- **H₀:** β = 0 (within the same size group and year, the denial probability is equal for E-3 and H-1B)
- **H₁:** β ≠ 0 (two-sided)
- **Test:** **linear probability model** (the same regression on a 0/1 outcome), employer-clustered SE.
- **Why it fits:** EDA shows employer size is the main confounder (Fig. 7). The linear probability model gives the difference directly in percentage points, which is how the MID is stated.
- **Other tests considered:**
  - *Logistic regression:* gives an odds ratio, not a difference in pp. Run as a check with the same cells and clustering.
  - *Cochran–Mantel–Haenszel test by size × year:* the classic stratified 2×2 test, but it assumes independent filings; the logit gives the same odds ratio with clustering.
  - *Chi-square on the raw 2×2 table:* ignores the size confounder the problem statement is about.
- **Checks:** logistic regression; bootstrap p-value; leaving out the largest employer.

## T4. E-3 vs H-1B denial rate, same employer × year (P2)

- **Cells:** employer × fiscal year. Only employers filing both visa types in the same year contribute (exploration: 6,309 employers).
- **H₀:** β = 0 (within the same employer and year, the denial probability is equal for E-3 and H-1B)
- **H₁:** β ≠ 0 (two-sided)
- **Test:** as T3, with the finer cells.
- **Why it fits:** it holds fixed everything about the employer (experience, lawyer, form-filling practice), which T3's size groups only approximate.
- **Estimand caveat:** employers that file both visas in one year are mostly mid-sized and large, where denials are rare, so this answers the question for those employers.
- **Other tests considered:**
  - *Cochran–Mantel–Haenszel test by employer × year:* the standard test for a 2×2 gap within many groups; gives an odds ratio, not pp. Run as a check.
  - *Logistic regression with employer dummies:* with 1,547-plus dummies and most employer-years having no denial, the odds ratios come out biased.
  - *Wilcoxon signed-rank on employer gaps:* most employers have a zero gap, and it would repeat the paired t.
- **Checks:** CMH by employer × year; paired t on employer gaps; bootstrap p-value; leaving out the largest employer.

---

## What is reported for every test
Test statistic, df, raw p-value, Holm-adjusted p-value, decision at family-wise α = 0.05, the estimate with a 95% CI (log points and %, or pp), the verdict from the CI against ±MID, the number of rows, cells and employers used, and the checks listed above.

**Bootstrap p-value:** 1,999 simulated datasets in which E-3 has no effect, flipping the sign of whole employers' contributions at random, so one employer's filings always move together. The p-value is the share of simulated t-values at least as large as the real one; it checks the main p-value without relying on the t table.

## Changes since the original plan
Only the checks changed: the TOST step for "negligible" was dropped (the CI alone decides, and no verdict depended on it); the conditional logit for T4 was replaced by CMH, because the software failed on employers with thousands of filings; Mann–Whitney, Wilcoxon and the T3 CMH were dropped as repeats; the employer-level paired t-tests for T2 and T4 were added after the run, so they aren't Holm-corrected.
