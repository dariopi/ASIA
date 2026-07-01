# Final Analysis

This folder contains scripts to analyse and visualise the 2-fold benchmark results.

Included files:

- `plot_results.py`: plots the `val_RMSE` evolution by reading `results.tsv`
- `analyze_best_test.py`: analyses test predictions saved in `checkpoints/` and produces metrics and plots

Examples:

```bash
python final_analysis/plot_results.py
python final_analysis/analyze_best_test.py
```

To compare the test RMSE against the benchmark workbook, copy `Benchmark Results.xlsx`
into this folder and re-run `analyze_best_test.py`.
