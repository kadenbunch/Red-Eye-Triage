# Released results

Copy these files from the run that produced the manuscript (the Colab `publication_outputs` folder) so reviewers can check every reported number without retraining:

| File | Supports |
|---|---|
| `data_manifest.csv` | image list and labels (1,304 images) |
| `split_manifest.csv` and `split_indices.npz` | the exact 85/15 train-pool / held-out split |
| `table1_overall_metrics_cv.csv` | Table 1 |
| `table2_tableS1_perclass_metrics_cv.csv` | Table 2, Supplementary Table S1 |
| `table3_tableS2_threshold_analysis_cv.csv` | Table 3, Supplementary Table S2 |
| `model_hyperparameters.csv` | Methods |
| `metrics_bootstrap_raw.json` | bootstrap replicates behind every CI |
| `<model>_oof_predictions.npz` | pooled out-of-fold probabilities |
| `<model>_heldout_predictions.npz` | held-out test probabilities |
| `confusion_matrix_<model>.csv` | Supplementary Figure S2 |
| `duplicates_near.csv`, `duplicates_train_test_leakage.csv` | duplicate audit |
| `llm_outputs/*.json` | Figure 3 generations (prompt, output, audit, model, timestamp) |
| `environment_pip_freeze.txt` | exact package versions |

Note: the original Colab run saved results under the file names `table_s1_model_hyperparameters.csv`, `table_s2_overall_metrics_cv.csv`, `table_s3_perclass_metrics.csv` and `table_s4_threshold_analysis.csv`, and did not save out-of-fold predictions. Re-running step 1 with the released split regenerates all files above.
