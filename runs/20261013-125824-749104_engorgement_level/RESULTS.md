# TickPredict — engorgement level results

Completed research baseline on Penn validation images.

Run: 20261013-125824-749104_engorgement_level. Supported labels: fully engorged, semi-engorged, unengorged.

| Result | Value |
|---|---:|
| Correct predictions | 74 of 110 |
| Overall accuracy | 67.27% |
| Balanced accuracy | 74.42% |
| Best saved epoch | 3 |
| Total epochs run | 7 |
| Protected test images used (recorded provenance) | 0 |

## Confusion matrix

Rows are recorded labels; columns are model predictions. Balanced accuracy is the mean of the per-class recalls below.

| Recorded label | Predicted fully engorged | Predicted semi-engorged | Predicted unengorged | Recall |
|---|---:|---:|---:|---:|
| fully engorged | 14 | 0 | 2 | 14/16 = 87.50% |
| semi-engorged | 10 | 40 | 18 | 40/68 = 58.82% |
| unengorged | 2 | 4 | 20 | 20/26 = 76.92% |

The model made 36 errors among 110 validation observations.

## Learning history

The saved model was selected at epoch 3 using validation balanced accuracy. Training ran for 7 epochs. The same validation observations were reused across epochs.

![Validation balanced accuracy by epoch](validation_curve.png)

## Individual errors

| Tick ID | Recorded label | Model prediction |
|---|---|---|
| 280623 | unengorged | semi-engorged |
| 281942 | semi-engorged | unengorged |
| 283062 | semi-engorged | unengorged |
| 283821 | unengorged | fully engorged |
| 284150 | unengorged | semi-engorged |
| 284510 | semi-engorged | fully engorged |
| 285002 | semi-engorged | fully engorged |
| 285562 | semi-engorged | unengorged |
| 285761 | semi-engorged | unengorged |
| 286015 | semi-engorged | unengorged |
| 286235 | semi-engorged | fully engorged |
| 286398 | semi-engorged | unengorged |
| 286514 | semi-engorged | unengorged |
| 286576 | semi-engorged | fully engorged |
| 286752 | semi-engorged | fully engorged |
| 287120 | semi-engorged | unengorged |
| 287439 | semi-engorged | fully engorged |
| 287679 | semi-engorged | unengorged |
| 288022 | semi-engorged | unengorged |
| 288031 | unengorged | semi-engorged |
| 288293 | semi-engorged | unengorged |
| 288297 | semi-engorged | fully engorged |
| 288552 | semi-engorged | fully engorged |
| 288817 | semi-engorged | unengorged |
| 288819 | fully engorged | unengorged |
| 289171 | semi-engorged | unengorged |
| 289267 | semi-engorged | unengorged |
| 289656 | semi-engorged | unengorged |
| 289797 | semi-engorged | unengorged |
| 290075 | semi-engorged | unengorged |
| 290148 | semi-engorged | unengorged |
| 290639 | semi-engorged | fully engorged |
| 290727 | unengorged | fully engorged |
| 291213 | unengorged | semi-engorged |
| 291327 | semi-engorged | fully engorged |
| 291891 | fully engorged | unengorged |

## Interpretation

These validation observations were used for checkpoint selection, so these figures are not independent test results. They describe engorgement level classification for the listed labels in this Penn cohort. These saved results do not evaluate smartphone photographs, unsupported classes, or human Lyme risk. Softmax model scores are uncalibrated and must not be read as validated probabilities of correctness. Person-level separation cannot be checked because person identifiers are unavailable.

The run provenance records 0 protected test images used. Generating this report reads saved JSON only and performs no model evaluation.

## Saved files

- best.pt: selected model checkpoint.
- validation.json: confusion matrix, recorded labels, predictions and uncalibrated model scores.
- history.json: results for each training epoch.
- provenance.json: cohort fingerprints, classes, software versions and configuration.
- completion.json: completion status and best recorded validation score.

This report was generated from the saved run; its checkpoints and JSON records were left unchanged.
