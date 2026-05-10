# Final Analysis

Questa cartella contiene gli script per analizzare e visualizzare i risultati del benchmark a 2 fold.

File inclusi:

- `plot_results.py`: genera il grafico dell'evoluzione di `val_RMSE` leggendo `results.tsv`
- `analyze_best_test.py`: analizza le predizioni di test salvate in `checkpoints/` e genera metriche e grafici

Esempi:

```bash
python final_analysis/plot_results.py
python final_analysis/analyze_best_test.py
```

Se vuoi confrontare il test RMSE con il workbook del benchmark, copia `Benchmark Results.xlsx`
in questa cartella e rilancia `analyze_best_test.py`.
