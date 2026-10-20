# Independent Service Audit

`audit_service.py` independently reads `trajectory.jsonl`, `events.jsonl`, `manifest.json`, and `evaluator_truth.json`. It does not import `mission.py` or production completion logic.

The audit only verifies service when the event stream explicitly records `abstract_strike_service_start` and `abstract_strike_complete` for the claimed actor and target. It checks public observation availability, 20-second continuous dwell, every intermediate `dt_s=1` trajectory sample, target-radius position, aircraft type and zone from trajectory rows, helicopter home-zone restriction, fault/energy/communication pause state, and one completion per `(branch, target)` pair. A service reset between start and completion invalidates that dwell. Missing actor, target, observation, active-service, pause, position, or timing evidence is `NOT_VERIFIED`; absent trajectory is `NOT_RUN`. Position proximity alone never establishes active service.

The current producer does not export service-start/reset events or an active service target field. Current traces therefore remain `NOT_VERIFIED` for a claimed completion until that evidence is added; this auditor does not infer it.

The existing attempt007 evidence has no `trajectory.jsonl` and uses `mode=search`, so it is expected to report `NOT_RUN`.
