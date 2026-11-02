# Track3 v5 候选（新网格 v8 场景；参考 = learning_v8_geom 自参考代理，非 GT）

代理 = 第 0 帧整体刚性对齐 + v8 4096 网格点误差（cm），与 #3/#4 同一函数，只换参考树。可入选 = 在 v8 场景中回放（轨迹与提交网格一致）。

| ep | 物体 | 静止(v8) | v4 回放 @v8参考 | v4 @v7参考(#4时) | 最佳旧ckpt重评(v8) | 最佳 lv8r | 入选 | 入选 MPPE | SP | 跟随 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | white_pot+white_pot_lid | 10.29 | 10.44 | 10.23 | 10.3 (learning_v7r@5999616) | 9.91 (lv8r@3999744) | lv8r@3999744 | 9.91 | ✗ | 0.44/0.05 |
| 3 | white_pot+wooden_spoon | 9.01 | 8.95 | 9.02 | 8.93 (learning_v7r@3999744) | 9.07 (lv8r@1999872) | reeval:learning_v7r@3999744 | 8.93 | ✗ | 0.14/0.0 |
| 5 | white_pot+wooden_spoon | 13.28 | 13.33 | 13.84 | 12.93 (learning_v6i@7999488) | 13.65 (lv8r@1999872) | reeval:learning_v6i@7999488 | 12.93 | ✗ | 0.0/0.0 |
| 6 | white_pot+wooden_spoon | 8.35 | 8.42 | 8.48 | 8.29 (learning_v7r@5999616) | 8.4 (lv8r@1999872) | reeval:learning_v7r@5999616 | 8.29 | ✗ | 0.11/0.0 |
| 8 | beige_cup+white_pot | 6.12 | 2.89 | 2.86 | 3.44 (learning_v7r@9999360) | 3.42 (lv8r@9999360) | lv8r@9999360 | 3.42 | ✓ | 0.52/0.19 |
| 10 | blue_cup+beige_cup | 9.43 | 8.22 | 7.49 | 9.77 (learning_fm_hc@9999360) | 11.57 (lv8r@3999744) | reeval:learning_fm_hc@9999360 | 9.77 | ✗ | 0.91/0.09 |
| 14 | blue_cup+plastic_dish_rack | 11.86 | 6.46 | 6.47 | 6.06 (learning_fm_hc@7999488) | 4.78 (lv8r@9999360) | lv8r@9999360 | 4.78 | ✗ | 0.08/– |
| 15 | plastic_dish_rack+wooden_spoon | 13.53 | 13.53 | 13.83 | 13.49 (learning_v7t@5999616) | 15.53 (lv8r@3999744) | reeval:learning_v7t@5999616 | 13.49 | ✗ | –/0.0 |
| 16 | plastic_dish_rack+wooden_spoon | 12.97 | 12.94 | 13.38 | 12.9 (learning_v7t@7999488) | 13.96 (lv8r@7999488) | reeval:learning_v7t@7999488 | 12.9 | ✗ | –/0.0 |
| 19 | mini_sweeper+mini_dust_pan | 7.95 | 10.68 | 8.56 | 9.23 (learning_v7r@1999872) | 10.06 (lv8r@7999488) | reeval:learning_v7r@1999872 | 9.23 | ✗ | 0.06/0.01 |
| 20 | blue_cup+water_pitcher | 10.0 | 10.63 | 9.86 | 10.34 (learning_fm_hc_ct@1999872) | 12.32 (lv8r@1999872) | reeval:learning_fm_hc_ct@1999872 | 10.34 | ✗ | 0.8/0.0 |
| 25 | wooden_piece_1+wooden_piece_2 | 16.3 | 14.7 | 14.97 | 15.01 (learning_v7r@1999872) | 15.33 (lv8r@1999872) | reeval:learning_v7r@1999872 | 15.01 | ✗ | 0.0/0.0 |
| 28 | plastic_dish_rack+white_pot | 15.78 | 11.49 | 12.48 | 12.64 (learning_v7r@9999360) | 14.12 (lv8r@7999488) | reeval:learning_v7r@9999360 | 12.64 | ✗ | –/0.03 |
| 30 | mini_sweeper+plastic_dish_rack | 5.4 | 7.56 | 7.58 | 5.61 (learning_v7t@9999360) | 1.62 (lv8r@7999488) | lv8r@7999488 | 1.62 | ✓ | 0.89/– |
| 32 | blue_cup+wooden_spoon | 10.36 | 11.58 | 11.93 | 12.45 (learning_v5@3999744) | 12.19 (lv8r@9999360) | lv8r@9999360 | 12.19 | ✗ | 0.24/0.0 |
| 33 | blue_cup+wooden_spoon | 11.83 | 14.07 | 14.4 | 13.73 (learning_v7r@5999616) | 14.07 (lv8r@1999872) | reeval:learning_v7r@5999616 | 13.73 | ✗ | 0.09/0.0 |
| 35 | wooden_spoon | 11.71 | 11.42 | 11.94 | 11.02 (learning_v5@2998272) | 11.67 (lv8r@3999744) | reeval:learning_v5@2998272 | 11.02 | ✗ | 0.01 |
| 36 | wooden_spoon | 17.42 | 13.47 | 14.0 | 13.22 (learning_v7r@9999360) | 13.09 (lv8r@9999360) | lv8r@9999360 | 13.09 | ✗ | 0.0 |
| 37 | wooden_spoon | 6.17 | 6.12 | 6.06 | 6.13 (learning_v6i@1999872) | 5.89 (lv8r@5999616) | lv8r@5999616 | 5.89 | ✓ | 0.0 |
| 38 | mini_dust_pan | 13.84 | 13.17 | 13.17 | 13.12 (learning_v3@4999168) | 13.8 (lv8r@1999872) | reeval:learning_v3@4999168 | 13.12 | ✗ | 0.02 |

就绪 20/20 段均值：静止 11.08，v4 回放@v8参考 10.5，入选 10.12；入选代理 SP 3/20
注：v4 回放是在 v7 场景（旧网格）里滚出来的，用 v8 参考打分只作对照，不可入选（网格与轨迹不一致）。
