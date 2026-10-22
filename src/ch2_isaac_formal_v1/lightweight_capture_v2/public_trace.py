"""Compact, public-only one-second trace and exact five-second samples."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _keyed_delta(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    return {"upsert": {key: value for key, value in current.items()
                       if key not in previous or previous[key] != value},
            "delete": sorted(set(previous) - set(current))}


def _without_geometry(task: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in task.items()
            if key not in {"public_xy_km", "public_bounds_km", "sweep_waypoints_km"}}


class PublicTraceWriter:
    def __init__(self, output_dir: str | Path, static_config: Mapping[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        static = dict(static_config)
        self.static_hash = hashlib.sha256(_canonical_bytes(static)).hexdigest()
        (self.output_dir / "static_config.json").write_bytes(_canonical_bytes(static) + b"\n")
        self._ticks = gzip.open(self.output_dir / "ticks.jsonl.gz", "wt", encoding="utf-8",
                                compresslevel=6, newline="\n")
        self._view = gzip.open(self.output_dir / "view_5s.jsonl.gz", "wt", encoding="utf-8",
                               compresslevel=6, newline="\n")
        self._audit = gzip.open(self.output_dir / "evaluation_audit.jsonl.gz", "wt", encoding="utf-8",
                                compresslevel=6, newline="\n")
        self.previous: dict[str, dict[str, Any]] = {"aircraft": {}, "public_tasks": {},
                                                    "observations": {}}
        self.count = 0

    def append(self, tick: Mapping[str, Any], *, hidden_audit: Mapping[str, Any]) -> None:
        tasks = {str(key): _without_geometry(value)
                 for key, value in tick["public_tasks"].items()}
        aircraft = {str(key): dict(value) for key, value in tick["aircraft"].items()}
        observations = {str(item.get("observation_id", item.get("target_id"))): dict(item)
                        for item in tick["public_observations"]}
        current = {"aircraft": aircraft, "public_tasks": tasks, "observations": observations}
        record = {"sim_time_s": int(tick["sim_time_s"]), "static_config_sha256": self.static_hash,
                  "deltas": {name: _keyed_delta(self.previous[name], values)
                             for name, values in current.items()},
                  "events": tick["events"], "decisions": tick["decisions"], "costs": tick["costs"]}
        self._ticks.write(_canonical_bytes(record).decode("utf-8") + "\n")
        if int(tick["sim_time_s"]) % 5 == 0 or tick["decisions"]:
            view = {"sim_time_s": int(tick["sim_time_s"]),
                    "state_sample_phase": "poststep",
                    "static_config_sha256": self.static_hash,
                    "aircraft": aircraft, "public_tasks": tasks,
                    "public_observations": tick["public_observations"],
                    "events": tick["events"], "decisions": tick["decisions"],
                    "costs": tick["costs"]}
            self._view.write(_canonical_bytes(view).decode("utf-8") + "\n")
        if hidden_audit.get("target_discovered_ids_by_aircraft"):
            audit = {"sim_time_s": int(tick["sim_time_s"]), **dict(hidden_audit)}
            self._audit.write(_canonical_bytes(audit).decode("utf-8") + "\n")
        self.previous = current
        self.count += 1

    def flush(self) -> None:
        for stream in (self._ticks, self._view, self._audit):
            stream.flush()

    def close(self) -> None:
        for stream in (self._ticks, self._view, self._audit):
            if not stream.closed:
                stream.close()

    def file_info(self) -> dict[str, Any]:
        self.close()
        result = {"schema": "public_trace_delta_v1", "static_config_sha256": self.static_hash,
                  "one_second_records": self.count}
        for name in ("static_config.json", "ticks.jsonl.gz", "view_5s.jsonl.gz",
                     "evaluation_audit.jsonl.gz"):
            path = self.output_dir / name
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        return result


class PublicTraceReader:
    def __init__(self, trace_dir: str | Path):
        self.trace_dir = Path(trace_dir)
        self.static_config = json.loads((self.trace_dir / "static_config.json").read_text("utf-8"))
        expected = hashlib.sha256(_canonical_bytes(self.static_config)).hexdigest()
        self.static_hash = expected

    def iter_records(self):
        tables: dict[str, dict[str, Any]] = {"aircraft": {}, "public_tasks": {},
                                             "observations": {}}
        with gzip.open(self.trace_dir / "ticks.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["static_config_sha256"] != self.static_hash:
                    raise ValueError("public tick references a different static configuration")
                for name, delta in record["deltas"].items():
                    table = tables[name]
                    for key in delta["delete"]:
                        table.pop(key, None)
                    table.update(delta["upsert"])
                tasks = {}
                for task_id, value in tables["public_tasks"].items():
                    tasks[task_id] = {**self.static_config["public_task_geometry"][task_id], **value}
                now = int(record["sim_time_s"])
                observations = [item for item in tables["observations"].values()
                                if float(item["available_at_sim_time_s"]) <= now]
                yield {"projection": {"sim_time_s": now,
                                       "aircraft": dict(tables["aircraft"]),
                                       "public_tasks": tasks,
                                       "public_observations": observations},
                       "events": record["events"], "decisions": record["decisions"],
                       "costs": record["costs"]}

    def iter_public_states(self):
        for record in self.iter_records():
            yield record["projection"]
