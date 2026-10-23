"""Bounded, outcome-independent capture of ordinary Isaac decision states."""
from __future__ import annotations

import copy
import hashlib
import json
import time


PUBLIC_OBSERVATION_FIELDS = ("target_id", "kind", "zone", "target_xy_km", "observed_at_sim_time_s",
                             "available_at_sim_time_s", "observation_id")


def priority_key(seed: int, decision_sequence: int, sim_time_s: int) -> str:
    token = f"{int(seed)}:{int(decision_sequence)}:{int(sim_time_s)}".encode("ascii")
    return hashlib.sha256(token).hexdigest()


def public_projection(sim_time_s, readbacks, energy_remaining, active_task_ids,
                      public_queues, observations, platform_status=None):
    """Project only current operational fields; caller supplies already-filtered tasks."""
    aircraft = {}
    for aircraft_id in sorted(readbacks):
        rb = readbacks[aircraft_id]
        aircraft[aircraft_id] = {
            **{key: copy.deepcopy(platform_status[aircraft_id][key])
               for key in ("kind", "home_zone", "available", "faulted", "safety_paused",
                           "communication_held") if platform_status and key in platform_status.get(aircraft_id, {})},
            "position_m": [float(v) for v in rb["position_m"]],
            "velocity_mps": [float(v) for v in rb["velocity_mps"]],
            "available_energy_wh": float(energy_remaining[aircraft_id]),
            "active_task_ids": list(active_task_ids.get(aircraft_id, ())),
            "queue_task_ids": list(public_queues.get(aircraft_id, ())),
        }
    public_observations = []
    for item in observations:
        public_observations.append({key: copy.deepcopy(item[key]) for key in PUBLIC_OBSERVATION_FIELDS
                                    if key in item})
    return {"sim_time_s": int(sim_time_s), "aircraft": aircraft,
            "public_observations": public_observations}


class StateSampler:
    def __init__(self, seed: int, limit: int = 8):
        if int(limit) < 1:
            raise ValueError("limit must be positive")
        self.seed = int(seed)
        self.limit = int(limit)
        self._seen_times = set()
        self._selected = []
        self.cpu_ns = 0
        self.wall_ns = 0

    @property
    def selected(self):
        return tuple(self._selected)

    def offer(self, decision_sequence, sim_time_s, ordinary_events, forced_at_time,
              state_factory, projection, backend_time_s=None,
              continuation_phase="post_public_updates_pre_candidate_generation_pre_step"):
        """Retain only eligible states that can enter deterministic hash top-k."""
        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        sim_time_s = int(sim_time_s)
        if not ordinary_events or forced_at_time or sim_time_s in self._seen_times:
            self.wall_ns += time.perf_counter_ns() - wall_start
            self.cpu_ns += time.process_time_ns() - cpu_start
            return False
        self._seen_times.add(sim_time_s)
        key = priority_key(self.seed, decision_sequence, sim_time_s)
        if len(self._selected) >= self.limit and key >= self._selected[-1]["priority"]:
            self.wall_ns += time.perf_counter_ns() - wall_start
            self.cpu_ns += time.process_time_ns() - cpu_start
            return False
        row = {"priority": key, "decision_sequence": int(decision_sequence),
               "sim_time_s": sim_time_s, "ordinary_events": sorted(set(ordinary_events)),
               "backend_time_s": (sim_time_s if backend_time_s is None else float(backend_time_s)),
               "continuation_phase": str(continuation_phase),
               "private_state": copy.deepcopy(state_factory()),
               "public_projection": copy.deepcopy(projection)}
        self._selected.append(row)
        self._selected.sort(key=lambda item: item["priority"])
        del self._selected[self.limit:]
        self.wall_ns += time.perf_counter_ns() - wall_start
        self.cpu_ns += time.process_time_ns() - cpu_start
        return True

    def write(self, output_dir):
        """Persist selected label snapshots under labels/snapshots."""
        from pathlib import Path
        import pickle

        output_dir = Path(output_dir)
        private_dir = output_dir / "snapshots"
        private_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for index, row in enumerate(sorted(self._selected, key=lambda item: item["sim_time_s"])):
            stem = f"state_{index:02d}_{row['sim_time_s']:06d}"
            private_path = private_dir / f"{stem}.pickle"
            with private_path.open("wb") as stream:
                pickle.dump(row["private_state"], stream, protocol=pickle.HIGHEST_PROTOCOL)
            public_path = private_dir / f"{stem}.public.json"
            public_bytes = (json.dumps(row["public_projection"], sort_keys=True,
                                       separators=(",", ":")) + "\n").encode("utf-8")
            public_path.write_bytes(public_bytes)
            private_sha = hashlib.sha256(private_path.read_bytes()).hexdigest()
            public_sha = hashlib.sha256(public_bytes).hexdigest()
            entries.append({"state_id": stem,
                             "sim_time_s": row["sim_time_s"],
                             "decision_time_s": row["sim_time_s"],
                             "backend_time_s": row["backend_time_s"],
                             "continuation_phase": row["continuation_phase"],
                             "decision_sequence": row["decision_sequence"],
                             "priority_sha256": row["priority"],
                             "private_file": str(Path("snapshots") / private_path.name),
                             "private_sha256": private_sha,
                             "public_file": str(Path("snapshots") / public_path.name),
                             "public_sha256": public_sha})
        index = {"schema": "isaac_roi_state_sampling_v1", "sampling_seed": self.seed,
                 "limit": self.limit, "selected_count": len(entries), "states": entries,
                 "sampler_offer_wall_ns": self.wall_ns,
                 "sampler_offer_cpu_ns": self.cpu_ns}
        index_bytes = (json.dumps(index, indent=2) + "\n").encode("utf-8")
        (output_dir / "state_index.json").write_bytes(index_bytes)
        (output_dir / "state_index.sha256").write_text(
            hashlib.sha256(index_bytes).hexdigest() + "  state_index.json\n", encoding="ascii")
        return index
