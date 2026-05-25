# Cluster Profiling Validation

This report extends the base cluster profiling with an aTLP-style summary.

## Most discriminative numeric variables

| Variable | Kruskal H | p-value | eta squared |
|---|---:|---:|---:|
| gcs_motor | 32327.00 | 0.000e+00 | 0.604 |
| gcs_eyes | 18727.00 | 0.000e+00 | 0.350 |
| gcs_score | 18161.65 | 0.000e+00 | 0.339 |
| glucose_max | 13056.68 | 0.000e+00 | 0.244 |
| apsiii | 12919.01 | 0.000e+00 | 0.241 |
| apsiii_prob | 12919.01 | 0.000e+00 | 0.241 |
| resp_rate_max | 2963.59 | 0.000e+00 | 0.055 |
| bun_max | 2117.41 | 0.000e+00 | 0.040 |
| temperature_max | 1983.70 | 0.000e+00 | 0.037 |
| creatinine_max | 1693.12 | 0.000e+00 | 0.032 |
| heart_rate_max | 1583.90 | 0.000e+00 | 0.030 |
| temp_score | 1466.80 | 0.000e+00 | 0.027 |

## Most discriminative categorical variables

| Variable | Chi2 | p-value | Cramers V |
|---|---:|---:|---:|
| gcs_unable | 27376.84 | 0.000e+00 | 0.715 |
| ethnicity | 1136.58 | 2.712e-228 | 0.072 |
| admission_type | 1019.76 | 1.176e-193 | 0.068 |
| marital_status | 976.74 | 1.067e-197 | 0.067 |
| gender | 207.73 | 8.173e-44 | 0.062 |
| admission_location | 799.36 | 6.163e-142 | 0.060 |

## Cluster-level interpretation

### cluster_0 | neurologic_inflammatory_high_severity

- Severity rank: 1
- Size: 4861 patients (9.09%)
- Snapshot: apsiii=89.09, admission_age=67.34, creatinine_max=1.73, bun_max=32.58, mbp_min=55.95, temperature_max=37.47, urineoutput=1603.67, gcs_score=39.95
- Top high features: gcs_score (+2.764), apsiii_prob (+2.145), apsiii (+1.983), sepsis3 (+0.651), temp_score (+0.390)
- Top low features: gcs_motor (-2.136), gcs_eyes (-1.717), gcs_verbal (-1.354), mbp_min (-0.309), temperature_min (-0.309)
- Salient high profile: gcs_score (48.00, red, high), apsiii_prob (0.41, red, medium), apsiii (86.00, red, high)
- Salient low profile: gcs_motor (1.00, red, medium), gcs_eyes (1.00, red, medium), urineoutput (1365.00, green, medium)
- Most uncertain cells: temp_score (U=2.27), creatinine_max (U=0.94), bun_max (U=0.79)

### cluster_1 | mixed_intermediate

- Severity rank: 2
- Size: 2800 patients (5.23%)
- Snapshot: apsiii=44.44, admission_age=65.02, creatinine_max=1.45, bun_max=24.19, mbp_min=54.89, temperature_max=37.27, urineoutput=1704.68, gcs_score=0.27
- Top high features: temp_score (+0.560), mbp_score (+0.329), glucose_score (+0.262), wbc_max (+0.242), uo_score (+0.207)
- Top low features: gcs_motor (-2.348), gcs_eyes (-1.873), gcs_verbal (-1.859), temperature_min (-0.497), gcs_score (-0.479)
- Salient high profile: admission_age (66.72, yellow, high), sodium_min (137.00, green, high), glucose_max (182.00, yellow, medium)
- Salient low profile: gcs_motor (1.00, red, medium), gcs_eyes (1.00, red, medium), gcs_score (0.00, green, low)
- Most uncertain cells: gcs_score (U=5.87), temp_score (U=2.07), apsiii_prob (U=1.24)

### cluster_4 | mixed_intermediate

- Severity rank: 3
- Size: 4208 patients (7.87%)
- Snapshot: apsiii=49.32, admission_age=64.14, creatinine_max=1.90, bun_max=36.85, mbp_min=61.01, temperature_max=37.31, urineoutput=2033.76, gcs_score=3.69
- Top high features: glucose_max (+2.203), glucose_score (+1.534), glucose_min (+1.267), sodium_score (+0.671), bun_max (+0.506)
- Top low features: sodium_min (-0.555), gcs_score (-0.200), uo_score (-0.068), temp_score (-0.065), mbp_score (-0.057)
- Salient high profile: glucose_max (351.00, red, medium), gcs_eyes (4.00, green, high), gcs_motor (6.00, green, high)
- Salient low profile: gcs_score (0.00, green, low), sodium_min (135.00, green, high), temp_score (0.00, green, low)
- Most uncertain cells: temp_score (U=3.52), gcs_score (U=1.89), creatinine_max (U=1.03)

### cluster_2 | mixed_lower_severity

- Severity rank: 4
- Size: 30666 patients (57.33%)
- Snapshot: apsiii=39.25, admission_age=66.08, creatinine_max=1.32, bun_max=24.88, mbp_min=60.61, temperature_max=37.24, urineoutput=1738.01, gcs_score=2.39
- Top high features: resp_rate_score (+0.453), gcs_motor (+0.397), gcs_verbal (+0.359), gcs_eyes (+0.339), resp_rate_max (+0.092)
- Top low features: gcs_score (-0.305), glucose_max (-0.281), apsiii_prob (-0.249), glucose_score (-0.225), apsiii (-0.221)
- Salient high profile: gcs_eyes (4.00, green, high), gcs_motor (6.00, green, high), sodium_min (138.00, green, high)
- Salient low profile: gcs_score (0.00, green, low), apsiii_prob (0.06, green, low), glucose_max (147.00, green, high)
- Most uncertain cells: temp_score (U=3.06), gcs_score (U=1.96), creatinine_max (U=0.95)

### cluster_3 | stable_low_severity

- Severity rank: 5
- Size: 10958 patients (20.48%)
- Snapshot: apsiii=36.34, admission_age=60.83, creatinine_max=1.48, bun_max=22.64, mbp_min=62.02, temperature_max=37.56, urineoutput=1923.54, gcs_score=4.03
- Top high features: temperature_max (+0.324), gcs_motor (+0.322), heart_rate_max (+0.219), hr_score (+0.209), heart_rate_min (+0.190)
- Top low features: resp_rate_score (-1.207), resp_rate_max (-0.433), apsiii (-0.350), apsiii_prob (-0.303), admission_age (-0.236)
- Salient high profile: gcs_eyes (4.00, green, high), gcs_motor (6.00, green, high), sodium_min (138.00, green, high)
- Salient low profile: resp_rate_max (24.00, yellow, high), gcs_score (0.00, green, low), apsiii (33.00, green, medium)
- Most uncertain cells: temp_score (U=2.90), gcs_score (U=1.63), creatinine_max (U=1.45)
