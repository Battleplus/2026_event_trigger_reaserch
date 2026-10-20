# Collection Analysis v1

`diagnose_batch.py` is a read-only diagnostic for one completed formal collection output. It reads the frozen registry, `batch_receipt.json`, scene receipts, representative state indexes, suffix receipts, raw trajectory/event files, and optional reduction outputs. It never launches the collector, evaluator, aggregate reducer, or model code.

Run it after the remote batch has produced a receipt:

```powershell
python diagnose_batch.py `
  --registry ..\formal_collection_v1\formal_batch_registry.json `
  --batch-receipt E:\path\to\formal-v1\batch_receipt.json `
  --output collection_analysis.json
```

Missing, failed, mismatched, or incomplete evidence is retained as `UNKNOWN` and is excluded from descriptive means and verified label counts. A suffix contributes only when its receipt, worker return code, ordinary-state interface, raw trajectory/events, reduction identity, and reduction validity all agree. The 30/30 readiness flag additionally requires a terminal batch receipt with one identity-matched terminal entry per registered scene and independently verified positive/nonpositive labels in both train and validation; failed scenes remain recorded and are not replaced or imputed.

This report is descriptive engineering diagnostics only. Inferential tests, training, threshold fitting, closed-loop evaluation, and world-model evaluation are explicitly `NOT_RUN`.
