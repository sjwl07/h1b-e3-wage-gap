# Power check (exploration-set SEs)

| test   | description                                                     |   rows used (explore) |   E-3 rows |   employers |   employers with E-3 |   SE explore |   SE confirm (expected) | MID    |   power at MID | MDE (80%)   |
|:-------|:----------------------------------------------------------------|----------------------:|-----------:|------------:|---------------------:|-------------:|------------------------:|:-------|---------------:|:------------|
| T1     | E-3 vs H-1B prevailing wage, same SOC × state × year            |               2085247 |      49749 |       92920 |                12917 |      0.00371 |                 0.00739 | 5%     |          1     | 2.5%        |
| T2     | E-3 vs H-1B prevailing wage, same employer × SOC × state × year |                363040 |      18496 |        2828 |                 2828 |      0.00433 |                 0.00864 | 5%     |          0.999 | 2.9%        |
| T3     | E-3 vs H-1B denial rate, same employer-size bin × year          |               3165069 |      64024 |      130875 |                15079 |      0.00166 |                 0.00331 | 0.5 pp |          0.162 | 1.11 pp     |
| T4     | E-3 vs H-1B denial rate, same employer × year                   |               1441530 |      39171 |        6309 |                 6309 |      0.00063 |                 0.00126 | 0.5 pp |          0.929 | 0.42 pp     |

alpha = 0.0125 (Holm worst case, 0.05/4). SEs are employer-clustered; confirm SE = explore SE × sqrt(132,485 / 33,314 employers) = × 1.99. MDE = smallest effect detectable with 80% power. Only SEs were read from the exploration fits, not the estimates.
