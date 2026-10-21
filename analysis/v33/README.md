# V3.3 Analysis Reproduction

The archived analyzer in `tools/v33_conclusion_analysis.py` uses modules from the frozen V3.3 release. Set `PYTHONPATH` to that release's `package/civilian` directory before invoking it; the archive root's historical `package` directory belongs to an older version. The analyzer checks each collection progress binding and commit marker against the imported V3.3 source and configuration.

From the archive root in PowerShell, point `$run` at a locally available completed V3.3 run directory. Raw shard traces are intentionally not included in this archive.

```powershell
$env:PYTHONPATH = (Resolve-Path .\releases\gain-gate-v3-3-civilian-search-rescue\package\civilian).Path
$run = "<local V3.3 run directory>"
python -B .\analysis\v33\tools\v33_conclusion_analysis.py --help
python -B -c "import json,v3_shards; print(json.dumps(v3_shards.current_binding(),sort_keys=True))"
python -B .\analysis\v33\tools\v33_conclusion_analysis.py $run --output-dir .\reports\v33-recomputed
```

Before analysis, compare the printed `current_binding()` object with the `binding` object in the run's `COLLECTION_PROGRESS.json`. The report in `reports/v33-conclusion-final-168/` is an aggregate analysis of all 168 committed scenarios; its `summary.json` does not include raw trajectories. The frozen data gate was not met, so this output does not support a gain-gate model or a safety approval.
