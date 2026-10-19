"""Durable, peer-authenticated worker phase transitions over a Unix socket."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import tempfile
import time

from protocol_contract import N2_STAGE_ORDER, validate_protocol_contract

if os.name == "posix":
    import resource
else:
    resource = None


MAX_MESSAGE_BYTES = 8192
DEFAULT_HANDSHAKE_TIMEOUT_SECONDS = 5.0
NUMERIC_TOLERANCE_SECONDS = 1e-6
def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _payload_hash(value):
    # Valid protocol requests contain only finite JSON numbers.
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _audit(path, event, **fields):
    if path is None:
        return
    from infra_io import durable_append_jsonl

    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [safe(item) for item in value]
        return value

    durable_append_jsonl(path, safe({"event": event, **fields}))


def _receive_json(channel):
    response = b""
    while not response.endswith(b"\n"):
        block = channel.recv(4096)
        if not block:
            raise RuntimeError("PHASE_ACCOUNTING_ACK_MISSING")
        response += block
        if len(response) > MAX_MESSAGE_BYTES:
            raise RuntimeError("PHASE_ACCOUNTING_ACK_TOO_LARGE")
    value = json.loads(response)
    if not isinstance(value, dict):
        raise RuntimeError("PHASE_ACCOUNTING_ACK_NOT_OBJECT")
    return value


def _send_json(channel, value):
    payload = _json_bytes(value) + b"\n"
    if len(payload) > MAX_MESSAGE_BYTES:
        raise RuntimeError("PHASE_ACCOUNTING_MESSAGE_TOO_LARGE")
    channel.sendall(payload)


def request_boundary(row, *, protocol_path=None,
                     timeout_seconds=DEFAULT_HANDSHAKE_TIMEOUT_SECONDS):
    """Complete one bounded two-step handshake; never silently bypass the server."""
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("PHASE_HANDSHAKE_TIMEOUT_INVALID")
    path = os.environ.get("W1_PHASE_ACCOUNTING_SOCKET")
    sequence = row.get("protocol_sequence")
    event = row.get("event")
    if (isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1
            or event not in {"stage_enter", "transition", "snapshot"}):
        raise RuntimeError("PHASE_PROTOCOL_REQUEST_IDENTITY_INVALID")
    payload = dict(row)
    payload_hash = _payload_hash(payload)
    envelope = {"action": "request", "sequence": sequence,
                "payload_sha256": payload_hash, "payload": payload}
    _audit(protocol_path, "request", sequence=sequence, payload_sha256=payload_hash,
           stage=payload.get("stage"), next_stage=payload.get("next_stage"),
           phase_event=event, payload=payload)
    if not path:
        _audit(protocol_path, "request_failed", sequence=sequence,
               payload_sha256=payload_hash, outcome="ambiguous",
               error="PHASE_ACCOUNTING_SOCKET_REQUIRED")
        raise RuntimeError("PHASE_ACCOUNTING_SOCKET_REQUIRED")

    deadline = time.monotonic() + float(timeout_seconds)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("PHASE_ACCOUNTING_TIMEOUT")
            channel.settimeout(remaining)
            channel.connect(path)
            _send_json(channel, envelope)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("PHASE_ACCOUNTING_TIMEOUT")
            channel.settimeout(remaining)
            accepted = _receive_json(channel)
            if accepted.get("status") == "rejected":
                _audit(protocol_path, "rejected", sequence=sequence,
                       payload_sha256=payload_hash, server_error=accepted.get("error"),
                       server_stage=accepted.get("stage"), transition_committed=False)
                raise _RejectedBoundary("PHASE_ACCOUNTING_REJECTED:" + str(accepted.get("error")))
            if accepted.get("sequence") != sequence or accepted.get("payload_sha256") != payload_hash:
                raise RuntimeError("PHASE_ACCOUNTING_ACCEPT_IDENTITY_MISMATCH")
            if accepted.get("status") != "accepted":
                raise RuntimeError("PHASE_ACCOUNTING_ACCEPT_STATUS_INVALID")
            _audit(protocol_path, "server_accepted", sequence=sequence,
                   payload_sha256=payload_hash, semantic_status="accepted_not_committed",
                   server_stage=accepted.get("stage"))

            confirm = {"action": "worker_confirm", "sequence": sequence,
                       "payload_sha256": payload_hash}
            _audit(protocol_path, "worker_confirm_sent", sequence=sequence,
                   payload_sha256=payload_hash, semantic_status="worker_confirmed_acceptance")
            _send_json(channel, confirm)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("PHASE_ACCOUNTING_TIMEOUT")
            channel.settimeout(remaining)
            committed = _receive_json(channel)
            if committed.get("status") == "rejected":
                _audit(protocol_path, "rejected", sequence=sequence,
                       payload_sha256=payload_hash, server_error=committed.get("error"),
                       server_stage=committed.get("stage"), transition_committed=False)
                raise _RejectedBoundary("PHASE_ACCOUNTING_REJECTED:" + str(committed.get("error")))
            if committed.get("sequence") != sequence or committed.get("payload_sha256") != payload_hash:
                raise RuntimeError("PHASE_ACCOUNTING_COMMIT_IDENTITY_MISMATCH")
            if committed.get("status") != "committed":
                raise RuntimeError("PHASE_ACCOUNTING_COMMIT_STATUS_INVALID")
            _audit(protocol_path, "worker_confirmed_commit", sequence=sequence,
                   payload_sha256=payload_hash, semantic_status="worker_received_commit_ack",
                   server_stage=committed.get("stage"))
            return {
                "status": "acknowledged",
                "sequence": sequence,
                "payload_sha256": payload_hash,
                "server_accepted": True,
                "worker_confirmed": True,
                "protocol_committed": True,
                "transition_committed": event == "transition",
                "idempotent_replay": bool(committed.get("idempotent_replay", False)),
            }
    except _RejectedBoundary:
        raise
    except BaseException as exc:
        _audit(protocol_path, "ambiguous", sequence=sequence,
               payload_sha256=payload_hash,
               phase_event=event,
               error=type(exc).__name__ + ": " + str(exc),
               stage=payload.get("stage"), next_stage=payload.get("next_stage"))
        raise RuntimeError("PHASE_ACCOUNTING_OUTCOME_AMBIGUOUS:" + str(sequence)) from exc


class _RejectedBoundary(RuntimeError):
    pass


class StageServer:
    def __init__(self, path, request, *, tree_reader, cpu_origin_seconds=0.0,
                 wall_origin_monotonic=None, journal_path=None):
        if (type(cpu_origin_seconds) not in (int, float)
                or not math.isfinite(cpu_origin_seconds) or cpu_origin_seconds < 0):
            raise ValueError("PHASE_CPU_ORIGIN_INVALID")
        self.requested_path = Path(path)
        self.request = request
        self.stage_order = tuple(request.get("stage_order", ()))
        self.initial_stage = request.get("initial_stage")
        validate_protocol_contract(
            package_release=request.get("package_release"),
            protocol_id=request.get("protocol_id"),
            stage_protocol=request.get("stage_protocol"),
            stage_order=self.stage_order,
        )
        if self.stage_order != N2_STAGE_ORDER or self.initial_stage != N2_STAGE_ORDER[0]:
            raise ValueError("PHASE_REQUEST_STAGE_ORDER_INVALID")
        if tuple(request.get("stages", {})) != self.stage_order:
            raise ValueError("PHASE_REQUEST_STAGE_CAPS_INVALID")
        self.tree_reader = tree_reader
        self.journal_path = (Path(journal_path) if journal_path is not None
                             else self.requested_path.parent / "supervisor-phase-protocol.jsonl")
        self.rows = []
        self.stage = self.initial_stage
        self.cpu_origin_seconds = float(cpu_origin_seconds)
        self.cpu_start = float(cpu_origin_seconds)
        self.wall_origin_monotonic = (time.monotonic() if wall_origin_monotonic is None
                                      else wall_origin_monotonic)
        if (type(self.wall_origin_monotonic) not in (int, float)
                or not math.isfinite(self.wall_origin_monotonic)
                or self.wall_origin_monotonic < 0):
            raise ValueError("PHASE_WALL_ORIGIN_INVALID")
        self.wall_start = float(self.wall_origin_monotonic)
        self.worker_initialized = False
        self.worker_phase_cpu_origin = 0.0
        self.last_worker_cpu_end = 0.0
        self.worker_phase_wall_origin = None
        self.last_worker_wall_end = None
        self.last_worker_sample_cpu_end = None
        self.last_worker_sample_wall_end = None
        self.last_cpu_cumulative = 0.0
        self.last_sequence = 0
        self.last_committed = None
        self.pending_request = None
        self.last_rejection = None
        self.committed_requests = {}
        self.runtime_dir = Path(tempfile.mkdtemp(prefix="w1p-", dir="/tmp"))
        self.path = self.runtime_dir / "p"
        self.socket = None
        self._closed = False
        bound = False
        try:
            os.chmod(self.runtime_dir, 0o700)
            if len(os.fsencode(self.path)) >= 108:
                raise OSError("PHASE_SOCKET_PATH_TOO_LONG")
            channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            channel.bind(str(self.path))
            bound = True
            os.chmod(self.path, 0o600)
            channel.listen(1)
            channel.setblocking(False)
            self.socket = channel
        except BaseException:
            if self.socket is not None:
                self.socket.close()
            elif "channel" in locals():
                channel.close()
            if bound:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
            try:
                self.runtime_dir.rmdir()
            except FileNotFoundError:
                pass
            raise

    def _record(self, event, **fields):
        _audit(self.journal_path, event, **fields)

    def _send_response(self, channel, value):
        _send_json(channel, value)

    def _reject(self, channel, sequence, payload_hash, error, *, payload=None):
        row = {
            "event": "reject",
            "status": "rejected",
            "sequence": sequence,
            "payload_sha256": payload_hash,
            "stage": self.stage,
            "next_stage": payload.get("next_stage") if isinstance(payload, dict) else None,
            "error": str(error),
            "transition_committed": False,
        }
        self.last_rejection = row
        self._record("reject", **{key: value for key, value in row.items() if key != "event"})
        response = {"status": "rejected", "sequence": sequence,
                    "payload_sha256": payload_hash, "error": str(error),
                    "stage": self.stage, "transition_committed": False}
        try:
            self._send_response(channel, response)
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            pass
        raise RuntimeError("PHASE_PROTOCOL_REJECTED:" + str(error))

    @staticmethod
    def _number(value, name):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise RuntimeError("PHASE_VALUE_INVALID:" + name)
        return float(value)

    @staticmethod
    def _payload_digest(payload):
        # Use allow_nan here so malformed numeric payloads get a durable, specific rejection.
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _validate_payload(self, payload, *, sequence, payload_hash, pid, worker_pid):
        if not isinstance(payload, dict):
            raise RuntimeError("PHASE_REQUEST_NOT_OBJECT")
        if payload.get("pid") != pid or type(payload.get("pid")) is not int:
            raise RuntimeError("PHASE_TRANSITION_IDENTITY_MISMATCH")
        if payload.get("protocol_sequence") != sequence:
            raise RuntimeError("PHASE_SEQUENCE_IDENTITY_MISMATCH")
        if type(payload.get("protocol_sequence")) is not int:
            raise RuntimeError("PHASE_SEQUENCE_IDENTITY_MISMATCH")
        if payload.get("event") not in {"stage_enter", "transition", "snapshot"}:
            raise RuntimeError("PHASE_EVENT_UNKNOWN")
        if payload.get("stage") != self.stage:
            raise RuntimeError("PHASE_TRANSITION_IDENTITY_MISMATCH")
        next_stage = payload.get("next_stage")
        if not isinstance(next_stage, str) or next_stage not in self.request["stages"]:
            raise RuntimeError("PHASE_NEXT_STAGE_UNKNOWN")
        event = payload["event"]
        if not self.worker_initialized:
            if (event != "stage_enter" or self.stage != self.initial_stage
                    or payload.get("stage") != self.initial_stage or next_stage != self.stage):
                raise RuntimeError("PHASE_STAGE_ENTER_REQUIRED")
        else:
            if event == "stage_enter":
                raise RuntimeError("PHASE_STAGE_ALREADY_ENTERED")
            if event == "transition" and next_stage == self.stage:
                raise RuntimeError("PHASE_TRANSITION_DID_NOT_CHANGE_STAGE")
            if event == "transition":
                current_index = self.stage_order.index(self.stage)
                if current_index + 1 >= len(self.stage_order) or next_stage != self.stage_order[current_index + 1]:
                    raise RuntimeError("PHASE_STAGE_TRANSITION_OUT_OF_ORDER")
            if event == "snapshot" and next_stage != self.stage:
                raise RuntimeError("PHASE_SNAPSHOT_CHANGED_STAGE")

        numeric = {}
        for name in ("cpu_start_seconds", "cpu_end_seconds", "cpu_seconds",
                     "wall_start_monotonic", "wall_end_monotonic", "wall_seconds"):
            numeric[name] = self._number(payload.get(name), name)
        if event == "stage_enter":
            numeric["stage_elapsed_cpu_seconds"] = self._number(
                payload.get("stage_elapsed_cpu_seconds"), "stage_elapsed_cpu_seconds"
            )
            numeric["stage_elapsed_wall_seconds"] = self._number(
                payload.get("stage_elapsed_wall_seconds"), "stage_elapsed_wall_seconds"
            )
        if numeric["cpu_end_seconds"] < numeric["cpu_start_seconds"]:
            raise RuntimeError("PHASE_CPU_CLOCK_REVERSED")
        if numeric["wall_end_monotonic"] < numeric["wall_start_monotonic"]:
            raise RuntimeError("PHASE_WALL_CLOCK_REVERSED")
        if event == "stage_enter":
            if abs(numeric["cpu_seconds"]) > NUMERIC_TOLERANCE_SECONDS:
                raise RuntimeError("PHASE_STAGE_ENTER_CPU_DELTA_NONZERO")
            if abs(numeric["wall_seconds"]) > NUMERIC_TOLERANCE_SECONDS:
                raise RuntimeError("PHASE_STAGE_ENTER_WALL_DELTA_NONZERO")
            if abs(numeric["stage_elapsed_cpu_seconds"] - (numeric["cpu_end_seconds"] - numeric["cpu_start_seconds"])) > NUMERIC_TOLERANCE_SECONDS:
                raise RuntimeError("PHASE_CPU_DELTA_MISMATCH")
            if abs(numeric["stage_elapsed_wall_seconds"] - (numeric["wall_end_monotonic"] - numeric["wall_start_monotonic"])) > NUMERIC_TOLERANCE_SECONDS:
                raise RuntimeError("PHASE_WALL_DELTA_MISMATCH")
        else:
            if abs(numeric["cpu_seconds"] - (numeric["cpu_end_seconds"] - numeric["cpu_start_seconds"])) > NUMERIC_TOLERANCE_SECONDS:
                raise RuntimeError("PHASE_CPU_DELTA_MISMATCH")
            if abs(numeric["wall_seconds"] - (numeric["wall_end_monotonic"] - numeric["wall_start_monotonic"])) > NUMERIC_TOLERANCE_SECONDS:
                raise RuntimeError("PHASE_WALL_DELTA_MISMATCH")
        expected_cpu_start = self.worker_phase_cpu_origin
        if abs(numeric["cpu_start_seconds"] - expected_cpu_start) > NUMERIC_TOLERANCE_SECONDS:
            raise RuntimeError("PHASE_CPU_BASELINE_DISCONTINUITY")
        if self.last_worker_wall_end is not None and numeric["wall_start_monotonic"] < self.last_worker_wall_end - NUMERIC_TOLERANCE_SECONDS:
            raise RuntimeError("PHASE_WALL_BASELINE_REVERSED")
        if event != "stage_enter" and self.worker_phase_wall_origin is not None and numeric["wall_start_monotonic"] < self.worker_phase_wall_origin - NUMERIC_TOLERANCE_SECONDS:
            raise RuntimeError("PHASE_WALL_BASELINE_REVERSED")
        if (self.last_worker_sample_cpu_end is not None
                and numeric["cpu_end_seconds"] < self.last_worker_sample_cpu_end - NUMERIC_TOLERANCE_SECONDS):
            raise RuntimeError("PHASE_CPU_SAMPLE_BASELINE_REVERSED")
        if (self.last_worker_sample_wall_end is not None
                and numeric["wall_end_monotonic"] < self.last_worker_sample_wall_end - NUMERIC_TOLERANCE_SECONDS):
            raise RuntimeError("PHASE_WALL_SAMPLE_BASELINE_REVERSED")
        if event == "stage_enter" and numeric["wall_start_monotonic"] < self.wall_origin_monotonic - NUMERIC_TOLERANCE_SECONDS:
            raise RuntimeError("PHASE_WALL_BASELINE_DISCONTINUITY")
        if numeric["wall_end_monotonic"] > time.monotonic() + 0.05:
            raise RuntimeError("PHASE_WALL_SNAPSHOT_IN_FUTURE")

        tree = self.tree_reader(os.getpid())
        if set(tree) != {os.getpid(), worker_pid}:
            raise RuntimeError("PHASE_BOUNDARY_HAS_ACTIVE_DESCENDANTS")
        return numeric

    def _make_row(self, payload, numeric):
        own = resource.getrusage(resource.RUSAGE_SELF)
        waited = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = own.ru_utime + own.ru_stime + waited.ru_utime + waited.ru_stime + numeric["cpu_end_seconds"]
        cumulative = cpu - self.cpu_origin_seconds
        if cumulative < self.last_cpu_cumulative - NUMERIC_TOLERANCE_SECONDS:
            raise RuntimeError("PHASE_CPU_CUMULATIVE_REVERSED")
        wall = time.monotonic()
        stage_elapsed_cpu = cpu - self.cpu_start
        stage_elapsed_wall = wall - self.wall_start
        if stage_elapsed_cpu < 0 or stage_elapsed_wall < 0:
            raise RuntimeError("PHASE_CLOCK_REVERSED")
        event = payload["event"]
        is_transition = event == "transition"
        row = {
            "event": event,
            "stage": payload["stage"],
            "next_stage": payload["next_stage"],
            "sequence": payload["protocol_sequence"],
            "payload_sha256": self._payload_digest(payload),
            "cpu_seconds": stage_elapsed_cpu if is_transition else 0.0,
            "wall_seconds": stage_elapsed_wall if is_transition else 0.0,
            "stage_elapsed_cpu_seconds": stage_elapsed_cpu,
            "stage_elapsed_wall_seconds": stage_elapsed_wall,
            "cpu_seconds_cumulative": cumulative,
            "sampled_monotonic": wall,
            "worker_reported_cpu_seconds": numeric["cpu_seconds"],
            "worker_reported_wall_seconds": numeric["wall_seconds"],
            "worker_stage_elapsed_cpu_seconds": numeric.get("stage_elapsed_cpu_seconds"),
            "worker_stage_elapsed_wall_seconds": numeric.get("stage_elapsed_wall_seconds"),
            "worker_cpu_end_seconds": numeric["cpu_end_seconds"],
            "worker_wall_end_monotonic": numeric["wall_end_monotonic"],
            "scope": "supervisor self and waited helpers plus quiescent worker self and waited descendants",
            "transition_committed": is_transition,
        }
        cap = self.request["stages"][payload["stage"]]
        prior_cpu = sum(item["cpu_seconds"] for item in self.rows
                        if item["stage"] == payload["stage"] and item["event"] == "transition")
        prior_wall = sum(item["wall_seconds"] for item in self.rows
                         if item["stage"] == payload["stage"] and item["event"] == "transition")
        if prior_cpu + stage_elapsed_cpu > cap["complete_process_cpu_seconds"]:
            raise RuntimeError("PHASE_RESOURCE_CAP_EXCEEDED:" + payload["stage"] + ":cpu")
        if prior_wall + stage_elapsed_wall > cap["wall_seconds"]:
            raise RuntimeError("PHASE_RESOURCE_CAP_EXCEEDED:" + payload["stage"] + ":wall")
        return row, cpu, wall, cumulative

    def _read_confirm(self, channel, sequence, payload_hash, *, duplicate):
        try:
            confirm = _receive_json(channel)
        except (socket.timeout, RuntimeError, ConnectionResetError) as exc:
            self._record("confirmation_ambiguous", sequence=sequence,
                         payload_sha256=payload_hash,
                         accepted_server_side=True,
                         error=type(exc).__name__ + ": " + str(exc),
                         stage=self.stage)
            return None
        if (confirm.get("action") != "worker_confirm"
                or confirm.get("sequence") != sequence
                or confirm.get("payload_sha256") != payload_hash):
            self._reject(channel, sequence, payload_hash,
                         "PHASE_WORKER_CONFIRM_IDENTITY_MISMATCH",
                         payload=self.pending_request.get("payload") if self.pending_request else None)
        self._record("worker_confirm", sequence=sequence,
                     payload_sha256=payload_hash,
                     idempotent_replay=duplicate,
                     semantic_status="worker_confirmed_acceptance")
        return confirm

    def process_pending(self, worker_pid):
        if self._closed or self.socket is None:
            raise RuntimeError("PHASE_SERVER_CLOSED")
        try:
            channel, _address = self.socket.accept()
        except BlockingIOError:
            return []
        with channel:
            channel.settimeout(DEFAULT_HANDSHAKE_TIMEOUT_SECONDS)
            sequence = None
            payload_hash = None
            payload = None
            try:
                pid, uid, _gid = struct.unpack(
                    "3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                )
                if pid != worker_pid or uid != os.getuid():
                    self._reject(channel, None, None, "PHASE_PEER_IDENTITY_MISMATCH")
                envelope = _receive_json(channel)
                if envelope.get("action") != "request":
                    self._reject(channel, envelope.get("sequence"), envelope.get("payload_sha256"),
                                 "PHASE_REQUEST_ACTION_INVALID")
                sequence = envelope.get("sequence")
                payload_hash = envelope.get("payload_sha256")
                payload = envelope.get("payload")
                self._record("request", sequence=sequence, payload_sha256=payload_hash,
                             stage=payload.get("stage") if isinstance(payload, dict) else None,
                             next_stage=payload.get("next_stage") if isinstance(payload, dict) else None,
                             phase_event=payload.get("event") if isinstance(payload, dict) else None,
                             payload=payload)
                if (type(sequence) is not int or sequence < 1
                        or not isinstance(payload_hash, str) or len(payload_hash) != 64):
                    self._reject(channel, sequence, payload_hash, "PHASE_SEQUENCE_INVALID", payload=payload)
                actual_hash = self._payload_digest(payload) if isinstance(payload, dict) else None
                if actual_hash != payload_hash:
                    self._reject(channel, sequence, payload_hash, "PHASE_PAYLOAD_DIGEST_MISMATCH", payload=payload)

                prior_committed = self.committed_requests.get(sequence)
                duplicate_committed = prior_committed is not None
                duplicate_pending = (self.pending_request is not None
                                     and sequence == self.pending_request["sequence"])
                if duplicate_committed:
                    prior = prior_committed
                    if payload_hash != prior["payload_sha256"] or payload != prior["payload"]:
                        self._reject(channel, sequence, payload_hash,
                                     "PHASE_SEQUENCE_PAYLOAD_CONFLICT", payload=payload)
                    self._record("duplicate_request", sequence=sequence,
                                 payload_sha256=payload_hash, idempotent=True,
                                 commit_already_durable=True)
                    self._send_response(channel, {"status": "accepted", "sequence": sequence,
                                                  "payload_sha256": payload_hash,
                                                  "stage": self.stage, "idempotent_replay": True})
                    self._read_confirm(channel, sequence, payload_hash, duplicate=True)
                    self._record("duplicate_commit", sequence=sequence,
                                 payload_sha256=payload_hash, idempotent=True)
                    try:
                        self._send_response(channel, {"status": "committed", "sequence": sequence,
                                                      "payload_sha256": payload_hash,
                                                      "stage": self.stage, "idempotent_replay": True})
                    except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError) as exc:
                        self._record("final_ack_ambiguous", sequence=sequence,
                                     payload_sha256=payload_hash,
                                     server_commit_durable=True,
                                     worker_may_not_have_received_commit=True,
                                     idempotent_replay=True,
                                     error=type(exc).__name__ + ": " + str(exc),
                                     stage=self.stage)
                    return []
                if duplicate_pending:
                    prior = self.pending_request
                    if payload_hash != prior["payload_sha256"] or payload != prior["payload"]:
                        self._reject(channel, sequence, payload_hash,
                                     "PHASE_SEQUENCE_PAYLOAD_CONFLICT", payload=payload)
                    pending = prior
                    duplicate = True
                    self._record("duplicate_request", sequence=sequence,
                                 payload_sha256=payload_hash, idempotent=True,
                                 awaiting_worker_confirmation=True)
                else:
                    if self.pending_request is not None:
                        self._reject(channel, sequence, payload_hash,
                                     "PHASE_PENDING_REQUEST_UNRESOLVED", payload=payload)
                    if sequence != self.last_sequence + 1:
                        self._reject(channel, sequence, payload_hash,
                                     "PHASE_SEQUENCE_INVALID", payload=payload)
                    numeric = self._validate_payload(payload, sequence=sequence,
                                                     payload_hash=payload_hash,
                                                     pid=pid, worker_pid=worker_pid)
                    pending = {"sequence": sequence, "payload_sha256": payload_hash,
                               "payload": payload, "numeric": numeric}
                    self.pending_request = pending
                    duplicate = False
                    self._record("accept", sequence=sequence,
                                 payload_sha256=payload_hash, stage=self.stage,
                                 next_stage=payload.get("next_stage"),
                                 phase_event=payload.get("event"),
                                 server_accepted=True, worker_confirmed=False,
                                 transition_committed=False)
                self._send_response(channel, {"status": "accepted", "sequence": sequence,
                                              "payload_sha256": payload_hash,
                                              "stage": self.stage,
                                              "idempotent_replay": duplicate})
                confirm = self._read_confirm(channel, sequence, payload_hash, duplicate=duplicate)
                if confirm is None:
                    return []

                try:
                    row, cpu, wall, cumulative = self._make_row(pending["payload"], pending["numeric"])
                except RuntimeError as exc:
                    self._reject(channel, sequence, payload_hash, str(exc), payload=pending["payload"])
                event = pending["payload"]["event"]
                worker_numbers = pending["numeric"]
                self._record("commit", sequence=sequence,
                             payload_sha256=payload_hash,
                             stage=row["stage"], next_stage=row["next_stage"],
                             phase_event=row["event"], server_accepted=True,
                             worker_confirmed=True, transition_committed=row["transition_committed"],
                             row=row)
                self.worker_initialized = True
                if event == "stage_enter":
                    self.worker_phase_cpu_origin = worker_numbers["cpu_start_seconds"]
                    self.last_worker_cpu_end = worker_numbers["cpu_start_seconds"]
                    self.worker_phase_wall_origin = worker_numbers["wall_start_monotonic"]
                    self.last_worker_wall_end = worker_numbers["wall_start_monotonic"]
                elif event == "transition":
                    self.worker_phase_cpu_origin = worker_numbers["cpu_end_seconds"]
                    self.last_worker_cpu_end = worker_numbers["cpu_end_seconds"]
                    self.worker_phase_wall_origin = worker_numbers["wall_end_monotonic"]
                    self.last_worker_wall_end = worker_numbers["wall_end_monotonic"]
                self.last_worker_sample_cpu_end = worker_numbers["cpu_end_seconds"]
                self.last_worker_sample_wall_end = worker_numbers["wall_end_monotonic"]
                self.last_cpu_cumulative = cumulative
                self.last_sequence = sequence
                self.last_committed = {"sequence": sequence,
                                       "payload_sha256": payload_hash,
                                       "payload": pending["payload"],
                                       "row": row}
                self.committed_requests[sequence] = self.last_committed
                self.pending_request = None
                if event == "transition":
                    self.rows.append(row)
                    self.stage = pending["payload"]["next_stage"]
                    self.cpu_start = cpu
                    self.wall_start = wall
                try:
                    self._send_response(channel, {"status": "committed", "sequence": sequence,
                                                  "payload_sha256": payload_hash,
                                                  "stage": self.stage,
                                                  "idempotent_replay": False})
                except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError) as exc:
                    self._record("final_ack_ambiguous", sequence=sequence,
                                 payload_sha256=payload_hash,
                                 server_commit_durable=True,
                                 worker_may_not_have_received_commit=True,
                                 error=type(exc).__name__ + ": " + str(exc),
                                 stage=self.stage)
                return [row]
            except _RejectedBoundary:
                raise
            except BaseException as exc:
                if self.pending_request is not None and self.pending_request.get("sequence") == sequence:
                    self._record("confirmation_ambiguous", sequence=sequence,
                                 payload_sha256=payload_hash,
                                 accepted_server_side=True,
                                 error=type(exc).__name__ + ": " + str(exc),
                                 stage=self.stage)
                    return []
                self._reject(channel, sequence, payload_hash, str(exc), payload=payload)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        try:
            if self.path.is_socket():
                self.path.unlink()
        except FileNotFoundError:
            pass
        try:
            self.runtime_dir.rmdir()
        except FileNotFoundError:
            pass
