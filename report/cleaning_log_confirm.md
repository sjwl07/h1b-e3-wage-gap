# Cleaning log (confirmation set, exploration-fitted rules)

| step                                                          |   rows |   rows_removed |   features |   cells_changed |   % missing cells |   % missing (excl. by design) |
|:--------------------------------------------------------------|-------:|---------------:|-----------:|----------------:|------------------:|------------------------------:|
| 0. raw (exploration-fitted rules applied)                     | 794670 |              0 |         17 |               0 |             10.65 |                          1.89 |
| 1. exact duplicates: KEEP, add n_identical                    | 794670 |              0 |         18 |               0 |             10.06 |                          1.79 |
| 2. normalize job_title text                                   | 794670 |              0 |         18 |          352727 |             10.06 |                          1.79 |
| 3. fill / fix employer_country                                | 794670 |              0 |         18 |           23725 |              9.9  |                          1.62 |
| 4. wage = 0 or no unit -> NaN                                 | 794670 |              0 |         18 |              26 |              9.9  |                          1.62 |
| 5. fix mislabelled pay units -> Year                          | 794670 |              0 |         19 |             148 |              9.37 |                          1.54 |
| 6. annualize wages                                            | 794670 |              0 |         21 |          794189 |             13.25 |                          1.39 |
| 7. harmonize SOC titles                                       | 794670 |              0 |         22 |           35339 |             12.64 |                          1.33 |
| 8. IQR outliers in log annual wage (within SOC) -> NaN + flag | 794670 |              0 |         26 |            7003 |             14.62 |                          1.2  |
| 9. flag missing worksite (not imputed)                        | 794670 |              0 |         27 |               0 |             14.07 |                          1.15 |
| 10. derived features                                          | 794670 |              0 |         40 |               0 |             10.23 |                          1.5  |
