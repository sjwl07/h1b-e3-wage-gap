# Questions and problem statements

Written 2026-10-21 after EDA on the exploration set (`report/eda.md`) and **before** the confirmation set was cleaned or looked at.

## 1. The obvious questions
These are what anyone would ask on first seeing six years of LCA filings. Each is answered by a count, a share or a grouped median, and most are already well known:

1. Which employers file the most? (Infosys, Tata Consultancy Services, Google, Amazon, Ernst & Young; the top 10 are 16% of all filings; Fig. 15)
2. Which jobs are filed for? (Mostly software: "Software Developers, Applications" alone is 30%, and computer titles together 65%; Fig. 4)
3. Where are the jobs? (California, Texas, New York lead; the top 15 states hold 83%; Fig. 3)
4. How many filings are certified? (91%; only 0.87% are denied; Fig. 6)
5. Have wages risen? (Median prevailing wage up 33% from FY2017 to FY2022; Fig. 2)
6. Are wages higher in some regions? (West $105k vs Midwest $78k; Fig. 3)
7. Do senior job titles get higher wages? (Yes, about +10% for the same job; Fig. 9)

## 2. Problem statements

**P1 (primary).** For the **same occupation, worksite state and fiscal year**, do E-3 (Australian) filings carry a different prevailing wage than H-1B filings? Does any difference remain when the comparison is **within the same employer**?

It could come out either way. The prevailing wage is set by DOL from the occupation, the area and the wage level (I–IV) the employer selects; the visa class isn't an input. So the null of no difference is a natural default. A difference in either direction would mean employers place the two groups at different wage levels. If the difference vanishes within employer, it's about *which* employers hire Australians; if it remains, it's about how the same employer files for the two groups.

**P2 (secondary).** Are E-3 filings **denied** more or less often than H-1B filings from **employers of the same size** in the same year, and within the **same employer**?

E-3 filings are denied 5.8× as often overall (Fig. 6), but half of them come from employers filing ≤10 times a year, who are denied far more often (Fig. 7). So the raw gap could be entirely employer mix (H₀) or partly a real difference.

## 3. Why these are not on the obvious list
- **They condition on what the obvious questions leave mixed.** Obvious questions 2, 3 and 5 show that occupation, place and year move the wage by tens of percent. A raw E-3 vs H-1B wage comparison would mostly measure that E-3 workers are in different jobs and states. P1 holds all three fixed, and then holds the employer fixed too.
- **They use the wage as a signal of the employer's choice.** The Kaggle extract has no wage-level column. Comparing prevailing wages within the same SOC × state × year recovers the level the employer chose indirectly; Fig. 9 checks that this works (senior titles +10%, junior −16%).
- **P2 is about a reversal risk (Simpson's paradox).** The obvious question 4 gives one denial rate. P2 asks whether a large raw gap between visa classes survives after employer size, which is the main confounder found in EDA.
- **Specific to this data:** P1 and P2 need visa class, SOC title, worksite, year, prevailing wage, case status and an employer identifier together, and E-3 is rarely studied next to H-1B.

## 4. Correlation is not causation
Both statements are worded as **differences between filings**, not effects of the visa.

- **P1, known confounders:** the worker's experience, education and seniority (not in the data; the job title only partly shows it), the exact metro area within a state (prevailing wages are set per metro area, and the state is coarser), and the employer's hiring policy. Australians may simply be hired into more senior roles.
- **P1, what the design addresses:** occupation, state and year are held fixed (T1), and then the employer too (T2). That rules out "high-paying employers hire Australians" as the whole explanation, but not differences between the workers themselves.
- **P2, known confounders:** employer experience with the process, use of an immigration lawyer, and errors on the form (denials are mostly for incomplete or inconsistent forms). The design holds employer size (T3) and then the employer itself (T4) fixed.
- **What a causal claim would need:** worker-level data (experience, degree, country of education) to compare similar workers; or a natural experiment, e.g. a rule change that affects only one visa, such as the wage-level-weighted H-1B lottery rule announced for the FY2022 cap and later vacated, studied before and after. Neither is in this data, so no causal claim is made.
