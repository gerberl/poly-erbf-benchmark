# Benchmark Summary (with TabPFN)

Loaded 494 results for 55 datasets (46 continuous, 9 discrete), 9 models

=== Performance Metrics (mean±std / median) ===
Note: Wins and Rank computed using R² adjusted for cross-dataset comparability
Model           N R²adjW  GapW   Rank    R² mean±std    med   R²adj mean±std    med Gap mean±std    med
------------------------------------------------------------------------------------------------------------------------
tabpfn         54     42     5   1.62    0.799±0.197  0.872      0.797±0.200  0.872  0.070±0.086  0.047
erbf           55      3     6   4.09    0.745±0.220  0.819      0.742±0.224  0.818  0.039±0.055  0.018
chebytree      55      2     9   4.15    0.736±0.227  0.832      0.734±0.230  0.831  0.041±0.055  0.026
xgb            55      1     1   4.44    0.738±0.222  0.828      0.735±0.225  0.828  0.107±0.107  0.079
chebypoly      55      1    16   4.85    0.710±0.226  0.792      0.708±0.228  0.791  0.029±0.040  0.014
ebm            55      3     7   5.13    0.701±0.219  0.752      0.698±0.222  0.752  0.036±0.046  0.020
rf             55      3     6   5.33    0.724±0.214  0.772      0.722±0.217  0.771  0.062±0.098  0.024
dt             55      0     5   7.45    0.671±0.237  0.718      0.668±0.241  0.716  0.051±0.052  0.032
ridge          55      0     0   7.95    0.579±0.229  0.601      0.576±0.231  0.597  0.011±0.015  0.004

=== Error Metrics (wins & mean rank - scale-free) ===
Note: MAE/RMSE values are scale-dependent; only ranks are meaningful across datasets
Model          MAE Wins   MAE Rank   RMSE Wins   RMSE Rank
------------------------------------------------------------
tabpfn               49       1.29          43        1.62
erbf                  0       4.07           3        4.05
chebytree             2       4.15           2        4.09
xgb                   1       4.27           1        4.42
chebypoly             1       5.13           1        4.93
ebm                   1       5.64           3        5.13
rf                    1       5.20           2        5.35
dt                    0       7.20           0        7.49
ridge                 0       8.05           0        7.93

=== Timing (per dataset avg / total) ===
Model                  Tune          Train           Eval          Total
----------------------------------------------------------------------
tabpfn            2.4s/2.2m      4.2s/3.8m      2.3m/2.1h      2.4m/2.2h
erbf            13.3m/12.2h    17.8s/16.3m      0.1s/7.0s    13.6m/12.5h
chebytree       59.5s/54.5m     0.5s/27.6s     0.2s/11.5s     1.0m/55.2m
xgb               3.2m/2.9h      1.2s/1.1m     0.2s/10.9s      3.2m/2.9h
chebypoly       38.2s/35.0m     0.4s/19.5s     0.2s/12.1s    38.7s/35.5m
ebm               7.2m/6.6h     10.5s/9.6m      0.0s/1.1s      7.4m/6.8h
rf              23.3m/21.3h    42.1s/38.6m      1.7s/1.5m    24.0m/22.0h
dt              28.0s/25.6m     0.3s/15.5s      0.0s/0.4s    28.3s/25.9m
ridge           19.2s/17.6m      0.0s/0.6s      0.0s/0.2s    19.3s/17.6m

=== Per-Instance Inference Time ===
Model           Mean (µs)  Median (µs)
--------------------------------------
tabpfn             1253.1        804.7
erbf                  2.1          1.9
chebytree             7.3          4.7
xgb                   3.4          2.0
chebypoly             3.5          2.2
ebm                   0.5          0.3
rf                   51.2         37.4
dt                    0.3          0.2
ridge                 0.2          0.1

=== Wins by Accuracy Metric ===
    model  R²  MAE  RMSE  Total
   tabpfn  42   49    43    134
      ebm   3    1     3      7
chebytree   2    2     2      6
     erbf   3    0     3      6
       rf   3    1     2      6
chebypoly   1    1     1      3
      xgb   1    1     1      3
       dt   0    0     0      0
    ridge   0    0     0      0

=== Stratum S1: Engineering/Simulation ===
Datasets: 13
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
tabpfn           11     5   1.38       0.943  0.967    0.006
erbf              1     3   3.31       0.917  0.948    0.003
chebytree         1     2   3.92       0.898  0.889    0.007
xgb               0     1   3.92       0.875  0.923    0.031
chebypoly         0     1   4.54       0.827  0.892    0.004
rf                0     0   6.08       0.870  0.883    0.017
ebm               0     1   6.23       0.796  0.870    0.009
dt                0     0   7.08       0.820  0.809    0.023
ridge             0     0   8.54       0.674  0.736    0.002

=== Stratum S2: Behavioural/Social ===
Datasets: 10
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
tabpfn            4     0   3.10       0.620  0.544    0.060
chebytree         1     1   3.80       0.557  0.475    0.033
ebm               2     1   4.30       0.528  0.480    0.035
chebypoly         1     5   4.30       0.550  0.480    0.034
rf                1     1   4.90       0.552  0.467    0.063
xgb               1     0   5.30       0.551  0.470    0.094
erbf              0     0   5.40       0.546  0.476    0.042
ridge             0     0   6.60       0.398  0.370    0.014
dt                0     2   7.30       0.498  0.420    0.059

=== Stratum S3: Physics/Chemistry/Life ===
Datasets: 16
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
tabpfn           14     0   1.25       0.723  0.756    0.080
erbf              1     0   3.81       0.658  0.676    0.047
chebypoly         0     5   4.50       0.630  0.655    0.035
chebytree         0     4   4.69       0.631  0.655    0.050
xgb               0     0   4.94       0.652  0.677    0.127
ebm               1     3   4.94       0.626  0.666    0.041
rf                0     3   5.06       0.636  0.663    0.056
ridge             0     0   7.50       0.545  0.535    0.011
dt                0     1   8.31       0.556  0.569    0.065

=== Stratum S4: Economic/Pricing ===
Datasets: 16
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
tabpfn           13     0   1.25       0.852  0.921    0.026
xgb               0     0   3.81       0.821  0.897    0.054
chebytree         0     2   4.00       0.813  0.876    0.018
erbf              1     3   4.19       0.809  0.874    0.014
ebm               0     2   4.94       0.797  0.848    0.012
rf                2     2   5.25       0.793  0.853    0.016
chebypoly         0     5   5.81       0.788  0.846    0.012
dt                0     2   7.00       0.764  0.813    0.023
ridge             0     0   8.75       0.636  0.681    0.003

=== Size: Small (<1K) (n < 1,000 samples) ===
Datasets: 17
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
tabpfn            9     0   1.88       0.668  0.687    0.060
erbf              2     1   4.18       0.640  0.661    0.042
chebytree         2     5   4.18       0.624  0.657    0.035
chebypoly         1     5   4.29       0.622  0.559    0.036
ebm               1     3   4.82       0.611  0.584    0.025
rf                2     0   4.88       0.637  0.621    0.066
ridge             0     0   6.47       0.542  0.551    0.014
xgb               0     1   6.53       0.616  0.656    0.103
dt                0     2   7.76       0.558  0.521    0.093

=== Size: Medium (1K-10K) (1,000 <= n < 10,000 samples) ===
Datasets: 19
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
tabpfn           17     5   1.32       0.868  0.945    0.019
erbf              0     0   3.79       0.823  0.896    0.019
xgb               0     0   4.05       0.800  0.904    0.058
rf                1     1   4.68       0.801  0.880    0.043
chebytree         0     3   4.74       0.804  0.865    0.018
chebypoly         0     6   4.89       0.758  0.846    0.014
ebm               1     2   5.68       0.742  0.850    0.023
dt                0     2   7.16       0.750  0.823    0.035
ridge             0     0   8.68       0.603  0.733    0.005

=== Size: Large (>=10K) (n >= 10,000 samples) ===
Datasets: 19
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
tabpfn           16     0   1.68       0.844  0.911    0.045
xgb               1     0   2.95       0.777  0.829    0.063
chebytree         0     1   3.53       0.762  0.841    0.025
erbf              1     5   4.32       0.753  0.818    0.008
ebm               1     2   4.84       0.732  0.772    0.016
chebypoly         0     5   5.32       0.734  0.819    0.011
rf                0     5   6.37       0.719  0.739    0.014
dt                0     1   7.47       0.684  0.716    0.017
ridge             0     0   8.53       0.578  0.645    0.001

=== Stratum x Size Matrix (R²adj winner) ===
Cell format: winner_model (wins/n_datasets)
Stratum  Small (<1K)          Medium (1K-10K)      Large (>=10K)       
----------------------------------------------------------------------
S1       chebytree (1/3)     tabpfn (8/8)        tabpfn (2/2)        
S2       chebypoly (1/5)     ebm (1/2)           tabpfn (2/3)        
S3       tabpfn (6/7)        tabpfn (5/5)        tabpfn (3/4)        
S4       rf (1/2)            tabpfn (3/4)        tabpfn (9/10)       

Dataset counts per cell:
  S1: 3, 8, 2
  S2: 5, 2, 3
  S3: 7, 5, 4
  S4: 2, 4, 10

=== Target Type: Continuous (Standard continuous regression targets) ===
Datasets: 46
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
tabpfn           38     5   1.46       0.837  0.916    0.042
erbf              3     6   3.93       0.778  0.845    0.018
xgb               1     1   4.20       0.770  0.839    0.070
chebytree         1     7   4.24       0.767  0.846    0.026
chebypoly         0    12   4.93       0.739  0.826    0.013
ebm               1     6   5.24       0.731  0.783    0.018
rf                2     5   5.41       0.752  0.821    0.023
dt                0     4   7.46       0.698  0.731    0.029
ridge             0     0   8.13       0.611  0.711    0.003

=== Target Type: Discrete (Discrete targets (counts, integers, ordinal ratings)) ===
Datasets: 9
Discrete datasets: Bike_Sharing_Demand, abalone, analcatdata_supreme, pmlb_1028_SWD (ordinal), pmlb_1029_LEV (ordinal), pmlb_1030_ERA (ordinal), pol (ordinal), student_performance (ordinal), wine_quality (ordinal)
Model        R²adjW  Gap*   Rank  R²adj mean    med  Gap med
--------------------------------------------------------------
tabpfn            4     0   2.44       0.596  0.544    0.052
chebytree         1     2   3.67       0.563  0.538    0.029
chebypoly         1     4   4.44       0.550  0.551    0.018
ebm               2     1   4.56       0.530  0.526    0.025
erbf              0     0   4.89       0.562  0.548    0.042
rf                1     1   4.89       0.564  0.524    0.063
xgb               0     0   5.67       0.557  0.501    0.092
ridge             0     0   7.00       0.394  0.388    0.013
dt                0     1   7.44       0.518  0.478    0.062

=== Dataset × Model: Adjusted R² ===
Rows sorted by stratum/dataset, columns by model ranking (best first)

Dataset                             Str    tabpfn      erbf chebytree       xgb chebypoly       ebm        rf        dt     ridge
---------------------------------------------------------------------------------------------------------------------------------
Ailerons                             S1     0.869     0.818     0.841     0.826     0.844     0.824     0.806     0.758     0.818
airfoil_noise                        S1     0.977     0.921     0.831     0.923     0.728     0.557     0.828     0.725     0.503
concrete_strength                    S1     0.945     0.885     0.865     0.917     0.892     0.907     0.880     0.809     0.597
cpu_act                              S1     0.988     0.983     0.982     0.982     0.983     0.981     0.976     0.968     0.736
elevators                            S1     0.929     0.862     0.886     0.873     0.890     0.873     0.689     0.594     0.812
energy_efficiency_heating            S1     0.998     0.998     0.997     0.996     0.997     0.989     0.995     0.996     0.913
feynman_gaussian                     S1     0.999     0.969     0.991     0.490     0.381     0.500     0.945     0.939     0.068
feynman_wave_interference            S1     0.999     0.965     0.883     0.935     0.909     0.846     0.888     0.873     0.841
friedman1                            S1     0.999     0.998     0.948     0.961     0.923     0.920     0.883     0.751     0.741
friedman1_d100                       S1     0.958     0.942     0.889     0.904     0.875     0.870     0.846     0.716     0.736
pmlb_215_2dplanes                    S1     0.948     0.948     0.948     0.947     0.948     0.705     0.947     0.948     0.705
pmlb_225_puma8NH                     S1     0.687     0.686     0.671     0.656     0.431     0.430     0.676     0.649     0.368
power_plant                          S1     0.967     0.948     0.944     0.961     0.941     0.952     0.947     0.939     0.928

Bike_Sharing_Demand                  S2     0.718     0.671     0.682     0.689     0.644     0.631     0.668     0.654     0.333
analcatdata_supreme                  S2     0.977     0.977     0.978     0.978     0.978     0.978     0.972     0.978     0.429
food_delivery_time                   S2       ---     0.252     0.335     0.346     0.331     0.336     0.284     0.278     0.181
pmlb_1028_SWD                        S2     0.412     0.405     0.412     0.387     0.409     0.415     0.409     0.363     0.388
pmlb_1029_LEV                        S2     0.544     0.548     0.538     0.501     0.551     0.545     0.524     0.478     0.551
pmlb_1030_ERA                        S2     0.352     0.344     0.365     0.325     0.340     0.356     0.342     0.334     0.351
pmlb_4544_GeographicalOriginalofMusic  S2     0.799     0.697     0.731     0.675     0.774     0.701     0.705     0.521     0.774
pol                                  S2     0.986     0.962     0.963     0.977     0.868     0.733     0.949     0.925     0.462
student_performance                  S2     0.275     0.212     0.232     0.195     0.239     0.231     0.275     0.147     0.228
wine_quality                         S2     0.514     0.387     0.335     0.439     0.363     0.353     0.391     0.301     0.287

abalone                              S3     0.586     0.556     0.564     0.521     0.559     0.526     0.548     0.481     0.514
diabetes                             S3     0.473     0.432     0.460     0.409     0.464     0.455     0.435     0.296     0.464
esol                                 S3     0.916     0.896     0.847     0.881     0.893     0.858     0.894     0.852     0.827
freesolv                             S3     0.935     0.885     0.884     0.882     0.886     0.898     0.877     0.782     0.823
lipophilicity                        S3     0.592     0.387     0.329     0.390     0.331     0.343     0.397     0.293     0.247
particulate-matter-ukair-2017        S3     0.744     0.740     0.746     0.741     0.750     0.752     0.726     0.718     0.733
physiochemical_protein               S3     0.768     0.505     0.505     0.615     0.404     0.378     0.424     0.359     0.281
pmlb_503_wind                        S3     0.815     0.791     0.785     0.768     0.799     0.784     0.760     0.706     0.758
pmlb_522_pm10                        S3     0.398     0.398     0.094     0.296     0.349     0.377     0.365     0.132     0.123
pmlb_529_pollen                      S3     0.794     0.791     0.792     0.740     0.791     0.782     0.710     0.639     0.792
pmlb_547_no2                         S3     0.600     0.588     0.543     0.519     0.539     0.555     0.599     0.469     0.476
qm7                                  S3     0.810     0.785     0.768     0.779     0.773     0.771     0.771     0.737     0.746
qsar_fish_toxicity                   S3     0.639     0.611     0.564     0.552     0.559     0.584     0.615     0.498     0.556
qsar_tid_11                          S3     0.650     0.526     0.535     0.607     0.398     0.315     0.468     0.393     0.317
sulfur                               S3     0.920     0.779     0.791     0.828     0.764     0.748     0.731     0.716     0.393
superconduct                         S3     0.931     0.848     0.890     0.909     0.819     0.888     0.856     0.819     0.675

Allstate_Claims_Severity             S4     0.538     0.494     0.500     0.480     0.470     0.459     0.438     0.401     0.450
Brazilian_houses                     S4     0.996     0.972     0.982     0.987     0.966     0.978     0.979     0.973     0.836
MiamiHousing2016                     S4     0.944     0.918     0.897     0.917     0.909     0.895     0.864     0.802     0.717
california_housing                   S4     0.875     0.792     0.751     0.829     0.730     0.772     0.739     0.667     0.601
diamonds                             S4     0.947     0.945     0.943     0.945     0.943     0.945     0.944     0.942     0.925
fiat_500_price                       S4     0.857     0.841     0.844     0.834     0.846     0.850     0.851     0.823     0.840
healthcare_insurance                 S4     0.856     0.839     0.856     0.845     0.833     0.737     0.855     0.840     0.733
house_16H                            S4     0.583     0.491     0.502     0.514     0.508     0.515     0.485     0.422     0.235
house_sales                          S4     0.903     0.870     0.852     0.876     0.846     0.847     0.838     0.790     0.746
medical_charges                      S4     0.980     0.980     0.980     0.979     0.980     0.980     0.977     0.978     0.827
nyc-taxi-green-dec-2016              S4     0.578     0.459     0.530     0.513     0.393     0.568     0.440     0.517     0.307
pmlb_218_house_8L                    S4     0.724     0.661     0.657     0.672     0.638     0.619     0.621     0.575     0.380
power_grid_stability                 S4     0.987     0.951     0.896     0.926     0.895     0.784     0.815     0.689     0.645
synthetic_multithreshold             S4     0.965     0.888     0.931     0.959     0.858     0.963     0.966     0.958     0.569
synthetic_piecewise                  S4     0.963     0.960     0.952     0.935     0.955     0.920     0.935     0.902     0.791
synthetic_step                       S4     0.939     0.877     0.932     0.922     0.844     0.923     0.939     0.938     0.579
