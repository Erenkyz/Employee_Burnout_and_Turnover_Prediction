# Observed stability and calibration results

Source: locally executed `scripts/evaluate_stability.py`, five seeds 42–46, each with stratified 882/294/294 train/validation/test split. The user reported 13/13 pytest tests passing on Windows / Python 3.13.

| Metric | Mean | Sample SD | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Average precision | 0.485529 | 0.038021 | 0.428903 | 0.523561 |
| ROC-AUC | 0.792196 | 0.022684 | 0.760789 | 0.821948 |
| Brier (lower better) | 0.184368 | 0.004451 | 0.178137 | 0.190604 |
| Precision | 0.422720 | 0.090440 | 0.274648 | 0.515152 |
| Recall | 0.582979 | 0.189469 | 0.361702 | 0.829787 |
| F1 | 0.464945 | 0.061606 | 0.412698 | 0.548673 |
| ECE (10 equal-width bins, lower better) | 0.245295 | 0.017577 | 0.216473 | 0.264134 |

Validation-selected thresholds by seed: 42: 0.63; 43: 0.63; 44: 0.67; 45: 0.71; 46: 0.42.

## Interpretation

Ranking quality (ROC-AUC) is less variable than threshold-sensitive recall and F1. High ECE and Brier relative to the prior-only dummy from the earlier single split suggest the class-weighted model's probability estimates need calibration assessment. These diagnostics do **not** demonstrate that calibration has been fixed.

## Limitations and next experiment

- Test subsets overlap between seeds, so the five runs are correlated and their standard deviations are descriptive, not confidence intervals.
- Equal-width-bin ECE depends on binning and sample size.
- Compare uncalibrated and calibrated probabilities with fitting and calibration restricted to training/validation partitions; reserve an untouched test partition for final evaluation.
- Do not choose the winner by repeatedly examining the same test data.
- Research portfolio only, not validated for employment decisions.
