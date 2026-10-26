# Pre-registered hypothesis tests

Fixed on 2026-10-21, **before** the confirmation set was cleaned or opened. The hypotheses came from EDA on the exploration set, so testing them on that same data would be circular. All four tests run once, on the held-out confirmation set (`data/raw/confirm.parquet`, 794,670 filings from 33,314 employers that never appear in the exploration set). It's cleaned with the rules fitted on the exploration set (`data/processed/clean_params.json`); nothing is re-fitted on it. The populations and models are defined in code in `src/tests_spec.py`, which `power.py` and `inference.py` both import.

## Common settings

| Item | Choice | Reason |
|---|---|---|
| Groups | E-3 Australian vs H-1B (variable `e3` = 1 for E-3) | H-1B1 (Chile/Singapore) is 0.4% of filings and a separate program. |
| Model | `y = β·e3 + cell fixed effect + error`, OLS; fixed effects removed by demeaning within cell | Fixed effects compare E-3 and H-1B filings **only within the same cell**, so differences between cells (occupation, state, year, employer) can't create a gap. β is the average of the within-cell gaps μ_E3 − μ_H1B (or p_E3 − p_H1B for denials), so β = 0 is the usual two-group null μ_E3 = μ_H1B, applied to like filings. |
| Standard errors | **Clustered by employer** (`employer_key`), CR1, t reference with G − 1 df | One employer's filings are near-copies (Fig. 12: 55% of rows at the largest filers have an identical twin), so they aren't independent. Clustering by employer is required, not optional; the split was grouped by employer for the same reason. |
| Significance | Family-wise α = 0.05, **Holm–Bonferroni** across the 4 tests | 4 tests on two related statements. Holm controls the family-wise error rate and is uniformly more powerful than Bonferroni. |
| Sidedness | **Two-sided** for all tests | EDA suggested directions (E-3 higher wage, more denials), but choosing one-sided tests after looking at the data is data snooping, and a reversal would be a finding in itself. |
| **MID, wage (T1, T2)** | **5%** in prevailing wage, i.e. \|β\| ≥ log(1.05) = 0.0488 | About $4,400 at the median wage ($88k); about one year of wage growth in this period (+33% over five years); about half of the gap between "senior" and other titles in the same job (Fig. 9). A smaller gap wouldn't put a worker at a different wage level. |
| **MID, denial (T3, T4)** | **0.5 percentage points** in denial probability | The H-1B denial rate is 0.78%, so +0.5 pp means a ≥1.6× higher risk. It's also more than half the entire FY2017–22 decline in the denial rate (1.36% → 0.49%), which was a notable policy-era change. |
| Estimate | β with 95% CI; wage results also as exp(β) − 1 (% difference) | Reported for every test, whether or not it rejects. |

**Interpreting each result:**

| Holm decision | 95% CI vs ±MID | Conclusion |
|---|---|---|
| Reject H₀ | entirely beyond ±MID | Statistically clear **and** meaningful |
| Reject H₀ | overlaps ±MID | Detectable, but possibly too small to matter |
| Not rejected | entirely inside ±MID | Practically negligible (confirmed with a TOST equivalence test at the same α) |
| Not rejected | crosses ±MID | **Inconclusive**: neither shown to matter nor shown negligible |

With ~800k filings, almost any difference will be statistically significant; the comparison of the CI with the MID is what decides whether it matters.

**Power** (`report/power.md`; employer-clustered SEs from the exploration set, scaled to the confirmation set's 33k employers, at α = 0.0125): T1 1.00, T2 0.999, T3 **0.16**, T4 0.93. T3 can only detect gaps of about 1.1 pp or more, so a non-rejection of T3 is read as *inconclusive*, never as "no difference".

---

## T1. E-3 vs H-1B prevailing wage, same occupation × state × year (P1)

- **Population:** E-3 and H-1B filings in FY2017–18 and FY2020–22 (FY2019 has no prevailing wage) with a cleaned wage and a known worksite.
- **Outcome:** log annual prevailing wage. **Cells:** SOC title × worksite state × fiscal year.
- **H₀:** μ_E3 = μ_H1B: within the same SOC × state × year, mean log PW is equal for E-3 and H-1B filings (β = 0)
- **H₁:** μ_E3 ≠ μ_H1B (β ≠ 0; two-sided)
- **Test:** Wald **t-test on β** in the fixed-effects model, employer-clustered SE, df = G − 1.
- **Why it fits:** the comparison has to hold occupation, place and year fixed (Figs. 2–4), and fixed effects do that for ~7,000 cells (with both visa types) without assuming anything about how cells differ. Log PW is symmetric (skew −0.04), and with thousands of clusters the clustered t is reliable.
- **Other tests considered:**
  - *Welch two-sample t-test on relative PW:* the same comparison, but it treats every filing as independent, which they aren't. It would understate the SE several-fold.
  - *Mann–Whitney U on relative PW:* tests a shift in ranks, which can't be compared with a 5% MID, and also assumes independence. Reported as a robustness check.
  - *Mixed model with employer random effects:* assumes employer effects are unrelated to visa class, which is exactly what T2 questions.
  - *Raw comparison without cells:* mixes occupation, state and year composition into the gap.
  - **Robustness checks reported:** restricted wild-cluster bootstrap p-value (Rademacher, 1,999 draws); the same model on the median-based relative PW from EDA; leaving out the largest employer.

## T2. The same comparison within the same employer (P1)

- **Cells:** employer × SOC title × worksite state × fiscal year. Only employers that file both E-3 and H-1B for the same job, state and year contribute (in exploration: 2,828 employers and ~18k E-3 filings).
- **H₀:** μ_E3 = μ_H1B: within the same employer, SOC, state and year, mean log PW is equal for E-3 and H-1B filings (β = 0)
- **H₁:** μ_E3 ≠ μ_H1B (β ≠ 0; two-sided)
- **Test:** as T1, with the finer cells.
- **Why it fits:** it removes everything about the employer (pay policy, industry, lawyer), so any gap left reflects how one employer files for the two groups. It's the H-1B counterpart of comparing homes within the same neighbourhood.
- **Estimand caveat:** it describes employers that hire both groups for the same job, which are larger than average; it isn't the average E-3 filing.
- **Other tests considered:**
  - *Paired t-test on employer-cell means:* similar, but it weights a cell with 1 E-3 and 500 H-1B filings the same as a balanced one, and pairing loses cells with unequal counts.
  - *Matching each E-3 filing to an H-1B filing of the same employer and cell:* a valid design that discards most H-1B filings for no gain in bias.

## T3. E-3 vs H-1B denial rate, same employer-size group × year (P2)

- **Population:** all E-3 and H-1B filings, FY2017–22 (case status is known for every year).
- **Outcome:** `denied` (1 = DENIED, else 0). **Cells:** employer-size group (filings per year: 1, 2–10, 11–100, 101–1000, > 1000) × fiscal year.
- **H₀:** p_E3 = p_H1B: within the same size group and year, the denial probability is equal for E-3 and H-1B (β = 0)
- **H₁:** p_E3 ≠ p_H1B (β ≠ 0; two-sided)
- **Test:** **linear probability model** with fixed effects, employer-clustered SE.
- **Why it fits:** EDA shows employer size is the main confounder (Fig. 7). The linear probability model gives the difference directly in percentage points, which is how the MID is stated; with a binary outcome, the clustered SE is valid without assuming a functional form.
- **Other tests considered:**
  - *Cochran–Mantel–Haenszel test stratified by size × year:* the classic stratified test of a 2×2 association, but it assumes independent filings. Reported as a robustness check.
  - *Logistic regression:* gives an odds ratio, not a difference in pp; at denial rates near 1% the two agree in sign and significance, so it's reported as a check.
  - *Chi-square on the raw 2×2 table:* ignores the size confounder that the problem statement is about.

## T4. E-3 vs H-1B denial rate, same employer × year (P2)

- **Cells:** employer × fiscal year. Only employers filing both visa types in the same year contribute (exploration: 6,309 employers).
- **H₀:** p_E3 = p_H1B: within the same employer and year, the denial probability is equal for E-3 and H-1B (β = 0)
- **H₁:** p_E3 ≠ p_H1B (β ≠ 0; two-sided)
- **Test:** as T3, with the finer cells.
- **Why it fits:** it holds fixed everything about the employer (experience, lawyer, form-filling practice), which T3's size groups only approximate.
- **Estimand caveat:** employers that file both visas in one year are mostly mid-sized and large, where denials are rare, so this answers the question for those employers.
- **Other tests considered:** *conditional (fixed-effects) logit:* drops every employer-year with no denials (most of them) and gives an odds ratio; reported as a check if feasible.

---

## What is reported for every test
Test statistic, df, raw p-value, Holm-adjusted p-value, decision at family-wise α = 0.05, the estimate with a 95% CI (log points and %, or pp), the position of that CI relative to ±MID, the TOST p-value, the number of rows, cells and employers used, and the robustness checks listed above.
