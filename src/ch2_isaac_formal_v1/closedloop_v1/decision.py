"""Lazy, public-state decision gating for the frozen ROI mission semantics."""
from __future__ import annotations

import time
from typing import Any, Callable


RISK_EVENTS = {"platform_fault", "communication_safety_hold", "energy_safety_hold"}
ORDINARY_EVENTS = {"task_activated", "public_observation", "search_cell_complete", "energy_anomaly"}


class WMHistoryUnavailable(RuntimeError):
    """Raised when the live public history cannot form a trained WM input window."""


class DecisionAdapter:
    """Gate planner invocation before candidate construction.

    ``planner`` must be the existing public replan catalog callable. The adapter
    passes its arguments through unchanged and does not inspect hidden outcomes.
    """

    MODES = {"existing_et", "periodic", "risk",
             "existing_et_gain_gate", "periodic_gain_gate"}

    def __init__(self, mode: str = "existing_et", periodic_interval_s: int = 240, event_threshold: int = 1,
                 event_cooldown_s: float = 240.0, risk_energy_fraction: float = 0.1,
                 gain_predictor: Callable[[dict[str, Any]], Any] | None = None,
                 gain_threshold: float = 0.0):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if periodic_interval_s <= 0:
            raise ValueError("periodic_interval_s must be positive")
        self.mode = mode
        self.periodic_interval_s = int(periodic_interval_s)
        self.event_threshold = int(event_threshold)
        self.event_cooldown_s = float(event_cooldown_s)
        self.risk_energy_fraction = float(risk_energy_fraction)
        self.gain_predictor = gain_predictor
        self.gain_threshold = float(gain_threshold)
        self.pending_events = 0
        self.last_event_open_s = float("-inf")
        self.last_periodic_s = 0

    def decide(self, backend, sim_time_s: int, projection: dict[str, Any],
               events: list[str], planner: Callable[..., tuple[list[dict], dict]],
               planner_args: tuple = (), planner_kwargs: dict | None = None,
               readback: dict[str, Any] | None = None) -> dict[str, Any]:
        """Use a live backend readback and invoke planner only after a gate opens."""
        planner_kwargs = planner_kwargs or {}
        readback_timing = None
        if readback is None:
            readback_wall, readback_cpu = time.perf_counter_ns(), time.process_time_ns()
            readback = backend.positions()
            readback_timing = {"wall_duration_ns": time.perf_counter_ns() - readback_wall,
                               "cpu_duration_ns": time.process_time_ns() - readback_cpu}
        if set(readback) != set(projection.get("aircraft", {})):
            raise ValueError("public projection aircraft IDs must match backend readback")
        event_set = set(events)
        forced_safety = bool(event_set & RISK_EVENTS)
        self.pending_events += sum(event in ORDINARY_EVENTS for event in events)
        low_energy = False
        feature_timing = None
        if self.mode == "risk" and not forced_safety:
            try:
                from ..light_models_v1.features import extract_features
            except ImportError:
                from light_models_v1.features import extract_features
            feature_wall, feature_cpu = time.perf_counter_ns(), time.process_time_ns()
            low_energy = extract_features(projection)["energy_norm_min"] <= self.risk_energy_fraction
            feature_timing = {"wall_duration_ns": time.perf_counter_ns() - feature_wall,
                              "cpu_duration_ns": time.process_time_ns() - feature_cpu}
        periodic = (int(sim_time_s) > 0 and int(sim_time_s) % self.periodic_interval_s == 0
                    and int(sim_time_s) != self.last_periodic_s)
        event_due = (self.mode in {"existing_et", "existing_et_gain_gate"} and
                     self.pending_events >= self.event_threshold and
                     sim_time_s - self.last_event_open_s >= self.event_cooldown_s)
        periodic_due = self.mode in {"periodic", "periodic_gain_gate"} and periodic
        risk_due = self.mode == "risk" and low_energy
        reasons = (["forced_safety"] if forced_safety else
                   [name for name, active in (("risk", risk_due), ("periodic", periodic_due),
                    ("Existing-ET", event_due)) if active])
        result: dict[str, Any] = {"sim_time_s": int(sim_time_s), "triggered": bool(reasons),
                                  "mode": self.mode,
                                  "trigger_reasons": reasons, "backend_readback": readback,
                                  "planner_called": False, "selected_action": None,
                                  "timing": {"scope": "partial_adapter_timing; excludes caller projection assembly and full controller path",
                                             "backend_readback": readback_timing,
                                             "risk_feature_extraction": feature_timing,
                                             "planner_wall_duration_ns": 0,
                                             "planner_cpu_duration_ns": 0}}
        if not reasons:
            return result
        if periodic_due:
            self.last_periodic_s = int(sim_time_s)
        if event_due:
            self.pending_events = 0
            self.last_event_open_s = float(sim_time_s)
        gain_gated_mode = self.mode.endswith("_gain_gate")
        if not forced_safety and gain_gated_mode:
            if self.gain_predictor is None:
                raise ValueError(f"mode {self.mode} requires gain_predictor")
            wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
            try:
                prediction = self.gain_predictor(projection)
            except WMHistoryUnavailable:
                result.update({"triggered": False, "gate_suppressed": True,
                               "gate_suppression_reason": "WM_HISTORY_UNAVAILABLE",
                               "predicted_gain": None})
                result["timing"]["predictor_wall_duration_ns"] = time.perf_counter_ns() - wall_start
                result["timing"]["predictor_cpu_duration_ns"] = time.process_time_ns() - cpu_start
                return result
            predicted_gain = float(getattr(prediction, "score", prediction))
            result["timing"]["predictor_wall_duration_ns"] = time.perf_counter_ns() - wall_start
            result["timing"]["predictor_cpu_duration_ns"] = time.process_time_ns() - cpu_start
            result["predicted_gain"] = predicted_gain
            if hasattr(prediction, "score"):
                result["prediction_metadata"] = {
                    key: getattr(prediction, key) for key in (
                        "gain_mean", "gain_lower_bound", "threshold", "score", "mode",
                        "version", "device", "integration_status", "timings_ms")
                    if hasattr(prediction, key)}
            if predicted_gain <= self.gain_threshold:
                result["triggered"] = False
                result["gate_suppressed"] = True
                return result
        wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
        candidates, selected = planner(*planner_args, **planner_kwargs)
        result["timing"]["planner_wall_duration_ns"] = time.perf_counter_ns() - wall_start
        result["timing"]["planner_cpu_duration_ns"] = time.process_time_ns() - cpu_start
        result.update({"planner_called": True, "candidate_count": len(candidates),
                       "selected_action": selected})
        return result


def apply_transfer(action: dict[str, Any], queues: dict[str, list[str]], routes: dict[str, dict],
                   cursors: dict[str, list[int]], completed: set[str]) -> bool:
    """Apply the existing catalog's sole non-NOOP action without moving active work."""
    if action.get("kind") != "uav_transfer":
        return False
    donor, recipient, task_id = (str(action[key]) for key in ("donor", "recipient", "task_id"))
    donor_index = int(cursors[donor][0])
    if task_id in completed or task_id not in queues[donor][donor_index + 1:]:
        return False
    queues[donor].remove(task_id)
    queues[recipient].append(task_id)
    routes[recipient][task_id] = routes[donor][task_id]
    return True
