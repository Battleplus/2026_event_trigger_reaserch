"""Lossless per-actor field-delta trajectory storage for suffix evaluation."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping


FORMAT = "lightweight_actor_delta_v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


class DeltaTrajectoryWriter:
    def __init__(self, path: str | Path, compresslevel: int = 6):
        self.path = Path(path)
        self._stream = gzip.open(self.path, "wt", encoding="utf-8",
                                 compresslevel=compresslevel, newline="\n")
        self._previous: dict[tuple[str, str], dict[str, Any]] = {}
        self.row_count = 0
        self.reconstructed_sha256 = hashlib.sha256()
        self._stream.write(_canonical({"format": FORMAT}).decode("utf-8") + "\n")

    def write(self, line: str) -> int:
        row = json.loads(line)
        branch = str(row["branch"])
        aircraft_id = str(row["aircraft_id"])
        key = (branch, aircraft_id)
        previous = self._previous.get(key, {})
        current = dict(row)
        changed = {name: value for name, value in current.items()
                   if name not in previous or previous[name] != value}
        removed = sorted(set(previous) - set(current))
        delta = {"branch": branch, "aircraft_id": aircraft_id,
                 "set": changed, "delete": removed}
        encoded = _canonical(delta) + b"\n"
        self._stream.write(encoded.decode("utf-8"))
        self._previous[key] = current
        self.reconstructed_sha256.update(_canonical(row) + b"\n")
        self.row_count += 1
        return len(line)

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def iter_lossless_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    previous: dict[tuple[str, str], dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header.get("format") != FORMAT:
            raise ValueError("unsupported compact trajectory format")
        for line in stream:
            item = json.loads(line)
            key = (str(item["branch"]), str(item["aircraft_id"]))
            row = dict(previous.get(key, {}))
            for name in item.get("delete", ()):
                row.pop(name, None)
            row.update(item["set"])
            previous[key] = row
            yield row


def compact_file_info(path: str | Path, row_count: int, row_sha256: str) -> dict[str, Any]:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"format": FORMAT, "stored_bytes": file_path.stat().st_size,
            "stored_sha256": digest.hexdigest(),
            "reconstructed_row_count": int(row_count),
            "reconstructed_rows_sha256": row_sha256}


def public_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Keep evaluator-visible event fields while dropping bulky/private model state."""
    omitted = {"candidates", "candidate_catalog", "catalog", "old_logprob", "value", "legal_mask",
               "action_logits", "action_probabilities", "logits", "probabilities",
               "hidden_state", "ppo_state"}

    def project(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): project(item) for key, item in value.items()
                    if str(key) not in omitted and not str(key).startswith("ppo_")}
        if isinstance(value, list):
            return [project(item) for item in value]
        return value

    return project(event)
