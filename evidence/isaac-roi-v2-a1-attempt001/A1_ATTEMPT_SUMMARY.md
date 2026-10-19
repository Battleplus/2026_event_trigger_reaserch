# A1 Attempt 001: Evidence Summary

The actual Isaac representative run completed 5400 simulated seconds and wrote 194400 trajectory rows; runtime exit was 0. Wall time was 101.61 seconds and maximum RSS was 4391696 KiB. The coordinator verified all six Isaac dependency hashes and confirmed the downloaded trajectory's uncompressed hash matched.

The independent audit failed with 506037 findings. Reported causes include actor type/kind/speed schema mismatch, incompatible actor/task types, and non-finite energy values. The maxspeed recheck is `NOT_VERIFIED` because the field was missing; the reported zero invariant failures do not establish acceptance. The coordinator decision is `REPAIR_REQUIRED`. The completed run is real simulator output, but it is not an accepted method result or a scientific success label. The paired A1 comparison, pilot, formal collection, and training remain NOT_RUN.

Large trajectory, event, and detailed audit payloads are not included. Their original relative-path hashes are retained in `OUTPUT_SHA256SUMS.txt`; resource figures are copied from the coordinator review's resource dictionary. This archive does not include the raw command-line resource log.
