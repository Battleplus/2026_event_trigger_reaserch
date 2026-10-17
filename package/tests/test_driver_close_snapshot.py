from __future__ import annotations

import sys
import shutil
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CIVILIAN_ROOT = PACKAGE_ROOT / "civilian"
sys.path.insert(0, str(CIVILIAN_ROOT))

import driver  # noqa: E402


class _WorkLedger:
    phase_protocol_sequence = ["stage-enter", "stage-exit"]

    def __init__(self) -> None:
        self.closed = False

    def work_snapshot(self) -> dict:
        if self.closed:
            raise RuntimeError("work ledger is closed")
        return {"pending_calls": 0, "failed_calls": 0, "totals": {"noop": {"reserved": 1, "actual": 1}}}

    def close(self) -> None:
        self.closed = True


class _Supervisor:
    def __init__(self, stage: str, ledger_path: Path, work_root: Path, export_root: Path, identity: str, **_kwargs) -> None:
        self.stage = stage
        self.runtime_identity_sha256 = identity
        self.work_ledger = _WorkLedger()
        self._closed = False
        self._stage_selected = False
        self._finished_row = None
        self._first_error = None

    def start_monitor(self) -> None:
        pass

    def select_stage(self, stage: str) -> None:
        self.stage = stage
        self._stage_selected = True
        self._finished_row = None

    def check(self, **_kwargs) -> None:
        pass

    def record_error(self, error) -> None:
        self._first_error = str(error)

    def finish(self, *, status: str, metadata: dict | None = None, **_kwargs) -> dict:
        self._finished_row = {"stage": self.stage, "status": status, "metadata": dict(metadata or {})}
        return dict(self._finished_row)

    def close(self, **_kwargs) -> dict:
        self.work_ledger.close()
        self._closed = True
        return dict(self._finished_row)


def test_worker_snapshots_settled_work_calls_before_closing_ledger(monkeypatch) -> None:
    tmp_path = Path(__file__).resolve().parents[2] / ".test-work-close-snapshot"
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir()
    monkeypatch.setattr(driver, "_read_prior_resource_cost", lambda *_args, **_kwargs: {"preflight_allowed": True})
    monkeypatch.setattr(driver, "ResourceSupervisor", _Supervisor)
    monkeypatch.setattr(driver, "_run_n2_stage", lambda *_args, **_kwargs: {"status": "accepted"})
    monkeypatch.setattr(driver, "ledger_summary", lambda *_args, **_kwargs: {"status": "accounted"})

    try:
        result = driver._execute_n2_worker(
            tmp_path,
            tmp_path,
            "a" * 64,
            invocation_start=(0.0, 0.0),
            full_chain=False,
        )

        assert result["status"] == "complete"
        assert result["work_calls"]["totals"]["noop"] == {"reserved": 1, "actual": 1}
        assert result["phase_protocol_sequence"] == ["stage-enter", "stage-exit"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
