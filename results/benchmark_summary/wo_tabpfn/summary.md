# Benchmark Summary (without TabPFN)

Loaded 440 results for 55 datasets (46 continuous, 9 discrete), 8 models

=== Performance Metrics (mean±std / median) ===
Note: Wins and Rank computed using R² adjusted for cross-dataset comparability
Model           N R²adjW  GapW   Rank    R² mean±std    med   R²adj mean±std    med Gap mean±std    med
------------------------------------------------------------------------------------------------------------------------
erbf           55     12     9   3.20    0.711±0.242  0.791      0.709±0.245  0.791  0.039±0.055  0.018
chebytree      55      5     9   3.45    0.702±0.250  0.785      0.699±0.254  0.785  0.041±0.054  0.029
xgb            55     16     1   3.58    0.705±0.245  0.771      0.702±0.248  0.770  0.107±0.108  0.085
chebypoly      55      8    19   3.91    0.680±0.244  0.764      0.678±0.247  0.764  0.029±0.040  0.014
ebm            55      6     7   4.09    0.674±0.239  0.748      0.672±0.242  0.748  0.035±0.044  0.020
rf             55      7     5   4.33    0.693±0.237  0.740      0.690±0.239  0.739  0.066±0.102  0.024
dt             55      0     5   6.49    0.645±0.255  0.706      0.642±0.258  0.706  0.049±0.051  0.032
ridge          55      1     0   6.95    0.557±0.240  0.581      0.554±0.242  0.579  0.011±0.016  0.003

=== Error Metrics (wins & mean rank - scale-free) ===
Note: MAE/RMSE values are scale-dependent; only ranks are meaningful across datasets
Model          MAE Wins   MAE Rank   RMSE Wins   RMSE Rank
------------------------------------------------------------
erbf                 13       3.15          12        3.15
chebytree             6       3.35           5        3.40
xgb                  21       3.35          18        3.53
chebypoly             4       4.15           6        4.00
ebm                   3       4.55           5        4.13
rf                    5       4.18           7        4.33
dt                    3       6.20           0        6.53
ridge                 0       7.09           2        6.95

=== Timing (per dataset avg / total) ===
Model                  Tune          Train           Eval          Total
----------------------------------------------------------------------
erbf              9.3m/8.6h    12.7s/11.6m      0.1s/5.5s      9.5m/8.7h
chebytree       46.2s/42.4m     0.4s/23.9s      0.2s/9.2s    46.9s/43.0m
xgb               2.5m/2.3h      1.1s/1.0m      0.2s/9.7s      2.5m/2.3h
chebypoly       25.8s/23.7m     0.3s/14.6s      0.1s/7.3s    26.2s/24.1m
ebm               5.5m/5.1h      8.4s/7.7m      0.0s/0.8s      5.7m/5.2h
rf              20.2m/18.5h    37.1s/34.0m      1.6s/1.4m    20.8m/19.1h
dt              19.0s/17.4m     0.2s/12.9s      0.0s/0.4s    19.2s/17.6m
ridge           11.3s/10.4m      0.0s/0.5s      0.0s/0.2s    11.4s/10.4m

=== Per-Instance Inference Time ===
Model           Mean (µs)  Median (µs)
--------------------------------------
erbf                  8.3          8.3
chebytree            31.7         18.7
xgb                  16.2          8.8
chebypoly            12.6         10.9
ebm                   1.5          1.4
rf                  203.9        141.3
dt                    1.3          0.7
ridge                 0.7          0.3

=== Wins by Accuracy Metric ===
    model  R²  MAE  RMSE  Total
      xgb  16   21    18     55
     erbf  12   13    12     37
       rf   7    5     7     19
chebypoly   8    4     6     18
chebytree   5    6     5     16
      ebm   6    3     5     14
       dt   0    3     0      3
    ridge   1    0     2      3

=== Stratum S1: Engineering/Simulation ===
Datasets: 13
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
erbf              4     6   2.69       0.859  0.948    0.003
xgb               3     1   3.00       0.820  0.923    0.031
chebytree         2     2   3.31       0.839  0.886    0.005
chebypoly         4     3   3.38       0.774  0.892    0.004
ebm               0     1   5.00       0.745  0.846    0.007
rf                0     0   5.00       0.819  0.883    0.014
dt                0     0   6.15       0.776  0.809    0.022
ridge             0     0   7.46       0.629  0.736    0.002

=== Stratum S2: Behavioural/Social ===
Datasets: 10
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
ebm               2     1   3.40       0.456  0.385    0.035
chebypoly         2     6   3.70       0.458  0.386    0.038
chebytree         1     1   3.70       0.456  0.389    0.033
rf                1     0   3.80       0.457  0.400    0.063
erbf              1     0   4.20       0.450  0.396    0.042
xgb               3     0   5.20       0.450  0.413    0.094
ridge             0     0   5.60       0.346  0.342    0.014
dt                0     2   6.40       0.418  0.348    0.059

=== Stratum S3: Physics/Chemistry/Life ===
Datasets: 16
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
erbf              3     0   2.88       0.650  0.676    0.047
chebypoly         2     5   3.56       0.627  0.655    0.025
chebytree         1     4   3.75       0.624  0.655    0.050
xgb               4     0   3.94       0.645  0.677    0.115
ebm               2     3   3.94       0.622  0.666    0.040
rf                3     3   4.06       0.630  0.663    0.054
ridge             1     0   6.56       0.541  0.535    0.006
dt                0     1   7.31       0.551  0.569    0.065

=== Stratum S4: Economic/Pricing ===
Datasets: 16
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
xgb               6     0   2.69       0.822  0.897    0.054
chebytree         1     2   3.12       0.812  0.876    0.018
erbf              4     3   3.31       0.808  0.874    0.014
ebm               2     2   3.94       0.797  0.848    0.012
rf                3     2   4.38       0.792  0.853    0.017
chebypoly         0     5   4.81       0.788  0.846    0.012
dt                0     2   6.00       0.763  0.813    0.023
ridge             0     0   7.75       0.636  0.681    0.003

=== Size: Small (<1K) (n < 1,000 samples) ===
Datasets: 17
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
erbf              3     1   3.29       0.631  0.611    0.042
chebypoly         4     5   3.59       0.609  0.559    0.040
chebytree         2     5   3.59       0.611  0.576    0.035
ebm               2     3   3.82       0.602  0.584    0.025
rf                4     0   3.94       0.626  0.615    0.066
ridge             1     0   5.41       0.531  0.551    0.014
xgb               1     1   5.59       0.605  0.561    0.103
dt                0     2   6.76       0.556  0.498    0.082

=== Size: Medium (1K-10K) (1,000 <= n < 10,000 samples) ===
Datasets: 19
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
erbf              5     3   2.95       0.781  0.885    0.023
xgb               5     0   3.16       0.760  0.881    0.058
rf                3     1   3.68       0.764  0.880    0.025
chebypoly         2     8   3.79       0.723  0.844    0.018
chebytree         3     3   4.00       0.761  0.856    0.018
ebm               1     2   4.53       0.707  0.846    0.023
dt                0     2   6.21       0.719  0.823    0.032
ridge             0     0   7.68       0.572  0.597    0.004

=== Size: Large (>=10K) (n >= 10,000 samples) ===
Datasets: 19
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
xgb              10     0   2.21       0.732  0.828    0.063
chebytree         0     1   2.79       0.716  0.778    0.025
erbf              4     5   3.37       0.707  0.779    0.008
ebm               3     2   3.89       0.699  0.765    0.012
chebypoly         2     6   4.32       0.694  0.764    0.009
rf                0     4   5.32       0.674  0.731    0.014
dt                0     1   6.53       0.642  0.689    0.017
ridge             0     0   7.58       0.557  0.645    0.001

=== Stratum x Size Matrix (R²adj winner) ===
Cell format: winner_model (wins/n_datasets)
Stratum  Small (<1K)          Medium (1K-10K)      Large (>=10K)       
----------------------------------------------------------------------
S1       erbf (2/3)          xgb (3/8)           chebypoly (2/2)     
S2       chebypoly (2/5)     ebm (1/2)           xgb (2/3)           
S3       chebypoly (2/7)     erbf (2/5)          xgb (3/4)           
S4       rf (1/2)            rf (2/4)            xgb (5/10)          

Dataset counts per cell:
  S1: 3, 8, 2
  S2: 5, 2, 3
  S3: 7, 5, 4
  S4: 2, 4, 10

=== Target Type: Continuous (Standard continuous regression targets) ===
Datasets: 46
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
erbf             11     9   3.13       0.756  0.840    0.018
xgb              14     1   3.20       0.750  0.831    0.076
chebytree         3     7   3.48       0.744  0.838    0.028
chebypoly         7    14   3.93       0.719  0.795    0.014
ebm               4     6   4.17       0.713  0.771    0.018
rf                6     5   4.43       0.734  0.793    0.023
dt                0     4   6.50       0.683  0.722    0.029
ridge             1     0   7.15       0.594  0.660    0.003

=== Target Type: Discrete (Discrete targets (counts, integers, ordinal ratings)) ===
Datasets: 9
Discrete datasets: Bike_Sharing_Demand, abalone, analcatdata_supreme, pmlb_1028_SWD (ordinal), pmlb_1029_LEV (ordinal), pmlb_1030_ERA (ordinal), pol (ordinal), student_performance (ordinal), wine_quality (ordinal)
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
chebytree         2     2   3.33       0.469  0.412    0.029
erbf              1     0   3.56       0.472  0.405    0.042
ebm               2     1   3.67       0.462  0.415    0.025
chebypoly         1     5   3.78       0.466  0.409    0.018
rf                1     0   3.78       0.470  0.409    0.063
xgb               2     0   5.56       0.457  0.439    0.092
ridge             0     0   5.89       0.353  0.351    0.013
dt                0     1   6.44       0.432  0.363    0.062

=== Dataset × Model: Adjusted R² ===
Rows sorted by stratum/dataset, columns by model ranking (best first)

Dataset                             Str      erbf chebytree       xgb chebypoly       ebm        rf        dt     ridge
-----------------------------------------------------------------------------------------------------------------------
Ailerons                             S1     0.758     0.778     0.770     0.780     0.765     0.756     0.718     0.756
airfoil_noise                        S1     0.921     0.831     0.923     0.728     0.557     0.828     0.725     0.503
concrete_strength                    S1     0.885     0.865     0.917     0.892     0.907     0.880     0.809     0.597
cpu_act                              S1     0.983     0.982     0.982     0.983     0.981     0.976     0.968     0.736
elevators                            S1     0.862     0.886     0.873     0.890     0.873     0.689     0.594     0.812
energy_efficiency_heating            S1     0.998     0.997     0.996     0.997     0.989     0.995     0.996     0.913
feynman_gaussian                     S1     0.969     0.991     0.490     0.381     0.500     0.945     0.939     0.068
feynman_wave_interference            S1     0.965     0.883     0.935     0.909     0.846     0.888     0.873     0.841
friedman1                            S1     0.998     0.948     0.961     0.923     0.920     0.883     0.751     0.741
friedman1_d100                       S1     0.244     0.183     0.249     0.257     0.255     0.237     0.180     0.211
pmlb_215_2dplanes                    S1     0.948     0.948     0.947     0.948     0.705     0.947     0.948     0.705
pmlb_225_puma8NH                     S1     0.686     0.671     0.656     0.431     0.430     0.676     0.649     0.368
power_plant                          S1     0.948     0.944     0.961     0.941     0.952     0.947     0.939     0.928

Bike_Sharing_Demand                  S2     0.671     0.682     0.689     0.644     0.631     0.668     0.654     0.333
analcatdata_supreme                  S2     0.977     0.978     0.978     0.978     0.978     0.972     0.978     0.429
food_delivery_time                   S2     0.252     0.335     0.346     0.331     0.336     0.284     0.278     0.181
pmlb_1028_SWD                        S2     0.405     0.412     0.387     0.409     0.415     0.409     0.363     0.388
pmlb_1029_LEV                        S2     0.548     0.538     0.501     0.551     0.545     0.524     0.478     0.551
pmlb_1030_ERA                        S2     0.344     0.365     0.325     0.340     0.356     0.342     0.334     0.351
pmlb_4544_GeographicalOriginalofMusic  S2     0.563     0.576     0.561     0.619     0.589     0.601     0.493     0.617
pol                                  S2     0.169     0.167     0.161     0.167     0.163     0.168     0.167     0.133
student_performance                  S2     0.188     0.177     0.116     0.180     0.194     0.205     0.130     0.188
wine_quality                         S2     0.387     0.335     0.439     0.363     0.353     0.391     0.301     0.287

abalone                              S3     0.556     0.564     0.521     0.559     0.526     0.548     0.481     0.514
diabetes                             S3     0.432     0.460     0.409     0.464     0.455     0.435     0.296     0.464
esol                                 S3     0.896     0.847     0.881     0.893     0.858     0.894     0.852     0.827
freesolv                             S3     0.885     0.884     0.882     0.886     0.898     0.877     0.782     0.823
lipophilicity                        S3     0.387     0.329     0.390     0.331     0.343     0.397     0.293     0.247
particulate-matter-ukair-2017        S3     0.740     0.746     0.741     0.750     0.752     0.726     0.718     0.733
physiochemical_protein               S3     0.505     0.505     0.615     0.404     0.378     0.424     0.359     0.281
pmlb_503_wind                        S3     0.791     0.785     0.768     0.799     0.784     0.760     0.706     0.758
pmlb_522_pm10                        S3     0.398     0.094     0.296     0.349     0.377     0.365     0.132     0.123
pmlb_529_pollen                      S3     0.791     0.792     0.740     0.791     0.782     0.710     0.639     0.792
pmlb_547_no2                         S3     0.588     0.543     0.519     0.539     0.555     0.599     0.469     0.476
qm7                                  S3     0.785     0.768     0.779     0.773     0.771     0.771     0.737     0.746
qsar_fish_toxicity                   S3     0.611     0.564     0.552     0.559     0.584     0.615     0.498     0.556
qsar_tid_11                          S3     0.419     0.427     0.491     0.350     0.251     0.376     0.324     0.251
sulfur                               S3     0.779     0.791     0.828     0.764     0.748     0.731     0.716     0.393
superconduct                         S3     0.842     0.888     0.914     0.819     0.887     0.856     0.820     0.674

Allstate_Claims_Severity             S4     0.480     0.487     0.491     0.462     0.457     0.429     0.395     0.447
Brazilian_houses                     S4     0.972     0.982     0.987     0.966     0.978     0.979     0.973     0.836
MiamiHousing2016                     S4     0.918     0.897     0.917     0.909     0.895     0.864     0.802     0.717
california_housing                   S4     0.792     0.751     0.829     0.730     0.772     0.739     0.667     0.601
diamonds                             S4     0.945     0.943     0.945     0.943     0.945     0.944     0.942     0.925
fiat_500_price                       S4     0.841     0.844     0.834     0.846     0.850     0.851     0.823     0.840
healthcare_insurance                 S4     0.839     0.856     0.845     0.833     0.737     0.855     0.840     0.733
house_16H                            S4     0.491     0.502     0.514     0.508     0.515     0.485     0.422     0.235
house_sales                          S4     0.870     0.852     0.876     0.846     0.847     0.838     0.790     0.746
medical_charges                      S4     0.980     0.980     0.979     0.980     0.980     0.977     0.978     0.827
nyc-taxi-green-dec-2016              S4     0.459     0.530     0.513     0.393     0.568     0.440     0.517     0.307
pmlb_218_house_8L                    S4     0.661     0.657     0.672     0.638     0.619     0.621     0.575     0.380
power_grid_stability                 S4     0.951     0.896     0.926     0.895     0.784     0.815     0.689     0.645
synthetic_multithreshold             S4     0.888     0.931     0.959     0.858     0.963     0.966     0.958     0.569
synthetic_piecewise                  S4     0.960     0.952     0.935     0.955     0.920     0.935     0.902     0.791
synthetic_step                       S4     0.877     0.932     0.922     0.844     0.923     0.939     0.938     0.579

=== Critical Difference Analysis (alpha=0.05) ===
Friedman test with Nemenyi post-hoc (autorank)

--- R² adjusted ---
Critical Difference (CD): 1.416
Rankings:
           meanrank    median       mad
ridge      6.945455  0.578571  0.198142
dt         6.490909  0.705557  0.224452
rf         4.327273  0.739469  0.195088
ebm        4.090909  0.748219  0.191201
chebypoly  3.909091  0.763839  0.184505
xgb        3.581818  0.770201  0.189075
chebytree  3.454545  0.784614  0.163766
erbf       3.200000  0.790602  0.174354
Plot: cd_plot_R2_adjusted.png

Statistical interpretation:
  The statistical analysis was conducted for 8 populations with 55 paired samples.
  The family-wise significance level of the tests is alpha=0.050.
  We rejected the null hypothesis that the population is normal for the populations ridge (p=0.000), rf (p=0.000), ebm (p=0.003), chebypoly (p=0.000), xgb (p=0.005), chebytree (p=0.000), and erbf (p=0.001). Therefore, we assume that not all populations are normal.
  Because we have more than two populations and the populations and some of them are not normal, we use the non-parametric Friedman test as omnibus test to determine if there are any significant differences between the median values of the populations. We use the post-hoc Nemenyi test to infer which differences are significant. We report the median (MD), the median absolute deviation (MAD) and the mean rank (MR) among all populations over the samples. Differences between populations are significant, if the difference of the mean rank is greater than the critical distance CD=1.416 of the Nemenyi test.
  We reject the null hypothesis (p=0.000) of the Friedman test that there is no difference in the central tendency of the populations ridge (MD=0.579+-0.220, MAD=0.198, MR=6.945), dt (MD=0.706+-0.240, MAD=0.224, MR=6.491), rf (MD=0.739+-0.247, MAD=0.195, MR=4.327), ebm (MD=0.748+-0.225, MAD=0.191, MR=4.091), chebypoly (MD=0.764+-0.239, MAD=0.185, MR=3.909), xgb (MD=0.770+-0.217, MAD=0.189, MR=3.582), chebytree (MD=0.785+-0.214, MAD=0.164, MR=3.455), and erbf (MD=0.791+-0.228, MAD=0.174, MR=3.200). Therefore, we assume that there is a statistically significant difference between the median values of the populations.
  Based on the post-hoc Nemenyi test, we assume that there are no significant differences within the following groups: ridge and dt; rf, ebm, chebypoly, xgb, chebytree, and erbf. All other differences are significant.

--- Generalization Gap ---
Critical Difference (CD): 1.416
Rankings:
           meanrank    median       mad
xgb        7.454545 -0.084830  0.046367
dt         5.454545 -0.031579  0.024936
rf         5.327273 -0.024437   0.01913
chebytree  4.563636 -0.029432  0.019471
ebm        4.127273 -0.019560  0.014258
erbf       3.981818 -0.018325  0.017377
chebypoly  3.309091 -0.014171  0.011609
ridge      1.781818 -0.003094  0.002884
Plot: cd_plot_Generalization_Gap.png

Statistical interpretation:
  The statistical analysis was conducted for 8 populations with 55 paired samples.
  The family-wise significance level of the tests is alpha=0.050.
  We rejected the null hypothesis that the population is normal for the populations xgb (p=0.000), dt (p=0.000), rf (p=0.000), chebytree (p=0.000), ebm (p=0.000), erbf (p=0.000), chebypoly (p=0.000), and ridge (p=0.000). Therefore, we assume that not all populations are normal.
  Because we have more than two populations and the populations and some of them are not normal, we use the non-parametric Friedman test as omnibus test to determine if there are any significant differences between the median values of the populations. We use the post-hoc Nemenyi test to infer which differences are significant. We report the median (MD), the median absolute deviation (MAD) and the mean rank (MR) among all populations over the samples. Differences between populations are significant, if the difference of the mean rank is greater than the critical distance CD=1.416 of the Nemenyi test.
  We reject the null hypothesis (p=0.000) of the Friedman test that there is no difference in the central tendency of the populations xgb (MD=-0.085+-0.044, MAD=0.046, MR=7.455), dt (MD=-0.032+-0.033, MAD=0.025, MR=5.455), rf (MD=-0.024+-0.027, MAD=0.019, MR=5.327), chebytree (MD=-0.029+-0.022, MAD=0.019, MR=4.564), ebm (MD=-0.020+-0.019, MAD=0.014, MR=4.127), erbf (MD=-0.018+-0.022, MAD=0.017, MR=3.982), chebypoly (MD=-0.014+-0.018, MAD=0.012, MR=3.309), and ridge (MD=-0.003+-0.007, MAD=0.003, MR=1.782). Therefore, we assume that there is a statistically significant difference between the median values of the populations.
  Based on the post-hoc Nemenyi test, we assume that there are no significant differences within the following groups: dt, rf, chebytree, and ebm; rf, chebytree, ebm, and erbf; chebytree, ebm, erbf, and chebypoly. All other differences are significant.
