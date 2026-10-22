"""Gated world-model entry point; G and G-WM remain explicit stubs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingReceipt:
    status: str
    stage: str
    device: str
    reasons: tuple[str, ...]
    updates: int
    checkpoint: str | None


def not_run_receipt(stage: str, device: str = "cuda:0") -> TrainingReceipt:
    return TrainingReceipt(
        status="NOT_RUN",
        stage=stage,
        device=device,
        reasons=(
            "TERMINAL_FORMAL_DATASET_NOT_SUPPLIED",
            "THIRTY_THIRTY_GATE_NOT_VERIFIED",
            "TRAINING_NOT_STARTED_IN_INTERFACE_PACKAGE",
        ),
        updates=0,
        checkpoint=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Conditional WM/G/G-WM receipt; no implicit training")
    parser.add_argument("--stage", choices=("world-model", "g", "g-wm"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output)
    if args.stage != "world-model":
        receipt = not_run_receipt(args.stage)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(receipt), indent=2, sort_keys=True), encoding="utf-8")
        return 0
    if not args.batch_output:
        parser.error("--batch-output is required for --stage world-model")
    from .trainer import run_training
    receipt = run_training(args.batch_output, output.parent,
                           max_epochs=args.epochs, batch_size=args.batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
