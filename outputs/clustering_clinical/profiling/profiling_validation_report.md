# Cluster Profiling Validation

This report extends the base cluster profiling with an aTLP-style summary.

## Most discriminative numeric variables

| Variable | Kruskal H | p-value | eta squared |
|---|---:|---:|---:|
| gcs_score | 31012.18 | 0.000e+00 | 0.580 |
| apsiii | 22940.66 | 0.000e+00 | 0.429 |
| bun_max | 14922.45 | 0.000e+00 | 0.279 |
| creatinine_max | 14767.58 | 0.000e+00 | 0.276 |
| sepsis3 | 5147.16 | 0.000e+00 | 0.096 |
| glucose_max | 2957.28 | 0.000e+00 | 0.055 |
| admission_age | 2062.05 | 0.000e+00 | 0.038 |
| mbp_min | 1622.15 | 0.000e+00 | 0.030 |
| mbp_max | 1495.96 | 0.000e+00 | 0.028 |
| sodium_min | 1339.15 | 1.079e-288 | 0.025 |
| urineoutput | 1319.92 | 1.596e-284 | 0.025 |
| temperature_max | 1176.13 | 2.387e-253 | 0.022 |

## Most discriminative categorical variables

| Variable | Chi2 | p-value | Cramers V |
|---|---:|---:|---:|
| admission_location | 17450.89 | 0.000e+00 | 0.285 |
| admission_type | 16987.26 | 0.000e+00 | 0.282 |
| gcs_unable | 2138.94 | 0.000e+00 | 0.200 |
| ethnicity | 1247.00 | 6.559e-252 | 0.076 |
| marital_status | 1227.63 | 1.749e-251 | 0.075 |
| gender | 37.62 | 1.341e-07 | 0.025 |

## Cluster-level interpretation

### cluster_0 | renal_metabolic_high_severity

- Severity rank: 1
- Size: 7463 patients (13.95%)
- Snapshot: apsiii=69.38, admission_age=65.15, creatinine_max=3.78, bun_max=57.10, mbp_min=55.34, temperature_max=37.25, urineoutput=1642.52, gcs_score=11.05
- Top high features: creatinine_max (+1.478), bun_max (+1.452), apsiii (+1.111), glucose_max (+0.940), sepsis3 (+0.405)
- Top low features: sodium_min (-0.439), mbp_min (-0.353), urineoutput (-0.117), temperature_max (-0.114), admission_age (+0.014)
- Salient high profile: bun_max (51.00, red, low), creatinine_max (2.90, red, low), apsiii (63.00, red, medium)
- Salient low profile: urineoutput (1125.72, red, low), sodium_min (135.00, green, high), temperature_max (37.11, yellow, high)
- Most uncertain cells: gcs_score (U=1.00), urineoutput (U=1.00), creatinine_max (U=0.84)

### cluster_4 | neurologic_inflammatory_high_severity

- Severity rank: 2
- Size: 8521 patients (15.93%)
- Snapshot: apsiii=59.56, admission_age=67.43, creatinine_max=1.05, bun_max=20.34, mbp_min=59.33, temperature_max=37.54, urineoutput=1709.68, gcs_score=21.52
- Top high features: gcs_score (+1.258), apsiii (+0.677), mbp_max (+0.596), sepsis3 (+0.354), temperature_max (+0.295)
- Top low features: bun_max (-0.266), creatinine_max (-0.250), glucose_max (-0.122), mbp_min (-0.063), urineoutput (-0.062)
- Salient high profile: gcs_score (15.00, red, low), apsiii (56.00, red, medium), admission_age (69.42, yellow, medium)
- Salient low profile: bun_max (18.00, yellow, medium), creatinine_max (0.90, yellow, medium), urineoutput (1525.00, yellow, low)
- Most uncertain cells: gcs_score (U=0.81), sepsis3 (U=0.76), urineoutput (U=0.59)

### cluster_2 | older_intermediate_moderate_severity

- Severity rank: 3
- Size: 8883 patients (16.61%)
- Snapshot: apsiii=46.16, admission_age=68.18, creatinine_max=1.30, bun_max=27.90, mbp_min=58.73, temperature_max=37.43, urineoutput=1780.18, gcs_score=6.07
- Top high features: admission_age (+0.189), sepsis3 (+0.172), temperature_max (+0.134), bun_max (+0.088), apsiii (+0.084)
- Top low features: glucose_max (-0.128), mbp_min (-0.107), sodium_min (-0.091), creatinine_max (-0.089), mbp_max (-0.061)
- Salient high profile: admission_age (71.14, yellow, medium), sodium_min (137.00, yellow, high), apsiii (44.00, yellow, medium)
- Salient low profile: glucose_max (154.00, yellow, medium), gcs_score (3.00, red, low), resp_rate_max (26.00, yellow, medium)
- Most uncertain cells: gcs_score (U=1.00), sepsis3 (U=0.94), urineoutput (U=0.64)

### cluster_1 | older_frail_lower_severity

- Severity rank: 4
- Size: 4490 patients (8.39%)
- Snapshot: apsiii=43.14, admission_age=70.85, creatinine_max=1.40, bun_max=30.31, mbp_min=60.04, temperature_max=37.22, urineoutput=1681.96, gcs_score=1.82
- Top high features: admission_age (+0.343), bun_max (+0.200), sepsis3 (+0.070), heart_rate_min (+0.022), resp_rate_max (+0.020)
- Top low features: gcs_score (-0.352), mbp_max (-0.189), temperature_max (-0.159), urineoutput (-0.085), glucose_max (-0.071)
- Salient high profile: admission_age (72.80, yellow, medium), sodium_min (138.00, yellow, high), bun_max (26.00, red, low)
- Salient low profile: temperature_max (37.11, yellow, high), urineoutput (1450.00, yellow, low), mbp_max (99.00, yellow, medium)
- Most uncertain cells: sepsis3 (U=0.95), gcs_score (U=0.80), urineoutput (U=0.65)

### cluster_3 | stable_low_severity

- Severity rank: 5
- Size: 24136 patients (45.12%)
- Snapshot: apsiii=30.57, admission_age=61.64, creatinine_max=0.92, bun_max=16.93, mbp_min=62.60, temperature_max=37.27, urineoutput=1877.32, gcs_score=0.00
- Top high features: mbp_min (+0.173), sodium_min (+0.110), urineoutput (+0.076), heart_rate_min (-0.073), temperature_max (-0.089)
- Top low features: apsiii (-0.605), gcs_score (-0.501), bun_max (-0.425), creatinine_max (-0.331), sepsis3 (-0.326)
- Salient high profile: sodium_min (138.00, yellow, high), mbp_min (62.00, yellow, medium), admission_age (63.53, yellow, medium)
- Salient low profile: apsiii (30.00, green, medium), gcs_score (0.00, green, low), bun_max (16.00, yellow, medium)
- Most uncertain cells: gcs_score (U=0.75), urineoutput (U=0.58), sepsis3 (U=0.56)
