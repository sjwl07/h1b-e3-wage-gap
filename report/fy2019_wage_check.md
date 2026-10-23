# FY2019 wage column: offered or prevailing?

FY2019 has `Wage_Rate_Of_Pay`; every other year has `Prevailing_Wage`. Annual-unit rows only.

|   fiscal_year | column           |   median_annual |   pct_multiple_of_1000 |
|--------------:|:-----------------|----------------:|-----------------------:|
|          2017 | Prevailing_Wage  |         72010.0 |                    0.7 |
|          2018 | Prevailing_Wage  |         82326.0 |                    0.9 |
|          2019 | Wage_Rate_Of_Pay |         92186.0 |                   39.4 |
|          2020 | Prevailing_Wage  |         91104.0 |                    0.6 |
|          2021 | Prevailing_Wage  |         95160.0 |                    1.0 |
|          2022 | Prevailing_Wage  |         95992.0 |                    0.6 |

Median year-on-year change of the per-SOC-title median wage:

|   from |     to |   common_soc_titles |   median_pct_change |
|-------:|-------:|--------------------:|--------------------:|
| 2017.0 | 2018.0 |               581.0 |                 6.8 |
| 2018.0 | 2019.0 |               564.0 |                15.6 |
| 2019.0 | 2020.0 |               502.0 |                -7.8 |
| 2020.0 | 2021.0 |               593.0 |                 4.0 |
| 2021.0 | 2022.0 |               616.0 |                 0.7 |

**Decision:** `Wage_Rate_Of_Pay` is the **offered wage**. About 40% of its values are whole thousands, against under 1% in every prevailing-wage year, and the per-SOC median jumps up into FY2019 and falls back out of it, while the other years move smoothly. It is kept as `wage_offered_fy2019`, and `prevailing_wage` is missing for FY2019 (missing by design). Prevailing-wage analyses therefore use FY2017-18 and FY2020-22.
