# Track3 #3 候选统一打分（参考 = learning_v7_geom 物体轨迹，自参考代理指标，非 GT）

| ep | 物体 | 静止 MPPE | #2 MPPE | 最佳候选 | 最佳 MPPE | 最佳 SP | 实际/参考位移 cm | 候选数 |
|---|---|---|---|---|---|---|---|---|
| 1 | white_pot+white_pot_lid | 10.32 | 25.5 | learning_v5@4999168 | 17.49 | ✗ | 33.1/22.8 vs 17.7/22.9 | 12 |
| 3 | white_pot+wooden_spoon | 9.07 | 14.93 | learning_v6i@9999360 | 9.22 | ✗ | 4.2/3.0 vs 7.0/28.5 | 16 |
| 5 | white_pot+wooden_spoon | 13.78 | 16.96 | learning_v6i@9999360 | 13.84 | ✗ | 1.3/0.2 vs 19.0/28.3 | 16 |
| 6 | white_pot+wooden_spoon | 8.41 | 15.04 | learning_v5@999424 | 9.11 | ✗ | 3.5/0.5 vs 11.3/24.4 | 12 |
| 8 | beige_cup+white_pot | 6.23 | 13.97 | learning_v7t@3999744 | 6.15 | ✗ | 0.9/1.3 vs 24.3/9.4 | 14 |
| 10 | blue_cup+beige_cup | 10.25 | 13.27 | learning_v3@2998272 | 10.2 | ✗ | 13.5/10.5 vs 6.8/28.9 | 11 |
| 14 | blue_cup+plastic_dish_rack | 11.8 | 12.68 | learning_v3@3999744 | 8.91 | ✗ | 20.5/5.2 vs 26.0/1.0 | 11 |
| 15 | plastic_dish_rack+wooden_spoon | 13.86 | 14.84 | learning_v7t@1999872 | 14.03 | ✗ | 1.4/0.1 vs 0.5/41.8 | 18 |
| 16 | plastic_dish_rack+wooden_spoon | 13.45 | 13.73 | sub2 | 13.73 | ✗ | 1.0/0.7 vs 2.4/36.7 | 12 |
| 19 | mini_sweeper+mini_dust_pan | 8.72 | 8.76 | learning_v5@1999872 | 8.56 | ✗ | 9.9/1.2 vs 13.7/15.3 | 11 |
| 20 | blue_cup+water_pitcher | 9.48 | 21.38 | learning_v5@3999744 | 15.88 | ✗ | 16.6/5.5 vs 7.8/28.4 | 11 |
| 25 | wooden_piece_1+wooden_piece_2 | 16.57 | 21.92 | learning_v5@1999872 | 14.97 | ✗ | 3.5/8.8 vs 33.9/22.1 | 11 |
| 28 | plastic_dish_rack+white_pot | 16.01 | 16.0 | sub2 | 16.0 | ✗ | 2.0/5.4 vs 1.8/51.0 | 11 |
| 30 | mini_sweeper+plastic_dish_rack | 7.47 | 8.84 | learning_v3@3999744 | 8.79 | ✗ | 10.5/3.4 vs 30.0/2.8 | 11 |
| 32 | blue_cup+wooden_spoon | 10.59 | 14.26 | learning_v3@4999168 | 11.93 | ✗ | 29.6/2.0 vs 14.0/30.5 | 12 |
| 33 | blue_cup+wooden_spoon | 12.01 | 12.39 | sub2 | 12.39 | ✗ | 1.4/2.2 vs 13.5/31.9 | 11 |
| 35 | wooden_spoon | 12.42 | 12.42 | learning_v5@2998272 | 11.94 | ✗ | 3.5 vs 20.5 | 16 |
| 36 | wooden_spoon | 18.16 | 18.23 | learning_v5@3999744 | 16.05 | ✗ | 9.8 vs 20.9 | 11 |
| 37 | wooden_spoon | 6.11 | 6.24 | learning_v6i@1999872 | 6.06 | ✓ | 0.1 vs 31.0 | 17 |
| 38 | mini_dust_pan | 13.84 | 14.04 | learning_v5@4999168 | 13.17 | ✗ | 2.9 vs 27.2 | 16 |

均值 MPPE 代理：静止 11.43，#2 14.77，#3 选择 11.92；SP 代理：#2 1/20，#3 1/20
