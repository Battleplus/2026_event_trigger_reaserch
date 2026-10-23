# 2026-10-23 collection correction

The user requires config.json once and records.jsonl with 5-second/actual
pre-decision samples, immediate events and a final summary. The old per-second
public tick, separate view and evaluation-audit files are not new routine outputs.
Old files remain preserved. Sampled counterfactual evidence stays under labels/;
this is not a second WM trajectory. WM transitions use actual sample delta time.

Live read-only SSH observation: server reachable, hostname ru-server-MS-1;
old collector PID 2570401 is stopped (T). No capture/suffix process active.
Original completed receipts: 92. Later lightweight continuation: 2 COMPLETE.
GPU1 has 40 MiB allocated; GPU0 is not used by this continuation.
There is no terminal union reduction or 30/30 result yet.

Local format/conversion checks: 7 passed, 5 deselected. This checks the writer
and legacy conversion, not Isaac operation or scientific validity. Legacy pair
evidence is referenced read-only rather than copying whole-scene raw traces.
Deployment and first new scene readback remain pending at this checkpoint.
Final update: first attempt was stopped because waypoint command lifecycle
events still produced unnecessary high-frequency records. Its evidence is
preserved as STOPPED_FOR_LOGGING_CORRECTION. The correction filters waypoint
traffic and records only the first lifecycle event per assignment/platform.
Eight targeted final tests passed. New version deployed on physical GPU1,
PID 4087734; 72 never-started registry scenes selected, no replacements or retries.
Live readback at 08:12:52 UTC: simulation time 4540 s, 921 samples including
periodic and actual decisions; 4 send/deliver/apply assignment events each;
no per-second ticks, standalone view or audit trajectory. Full scene/pair
completion remains pending. Original 94 completed scene receipts and two
interrupted scene directories remain preserved. No final 30/30 conclusion.
