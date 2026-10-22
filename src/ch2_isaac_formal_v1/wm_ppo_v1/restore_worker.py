"""Independent-process checkpoint restore probe for the world-model stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import restore_training_checkpoint
from .world_model import EventAwareWorldModel


def restore_probe(checkpoint: str | Path, device: str = "cpu") -> dict:
    import torch

    with Path(checkpoint).open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if payload.get("schema") != "wm_training_checkpoint_v2":
        raise ValueError("unsupported world-model checkpoint schema")
    architecture = metadata.get("architecture", {})
    required_architecture = {"node_dim", "hidden_dim", "dynamic_dim", "event_dim", "action_dim"}
    if not required_architecture.issubset(architecture):
        raise ValueError("checkpoint architecture metadata is incomplete")
    if metadata.get("feature_dim") != architecture["node_dim"] or metadata.get("event_dim") != architecture["event_dim"]:
        raise ValueError("checkpoint schema dimensions do not match architecture")
    model = EventAwareWorldModel(
        node_dim=int(architecture["node_dim"]), hidden_dim=int(architecture["hidden_dim"]),
        dynamic_dim=int(architecture["dynamic_dim"]), event_dim=int(architecture["event_dim"]),
        action_dim=int(architecture["action_dim"]),
    ).to(device)
    optimizer_spec = metadata.get("optimizer", {})
    if optimizer_spec.get("name") != "Adam" or "lr" not in optimizer_spec or "weight_decay" not in optimizer_spec:
        raise ValueError("checkpoint optimizer specification is incomplete")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(optimizer_spec["lr"]),
                                 weight_decay=float(optimizer_spec["weight_decay"]))
    return restore_training_checkpoint(checkpoint, model, optimizer, device=device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    receipt = restore_probe(args.checkpoint, args.device)
    print(json.dumps(receipt, sort_keys=True, default=str))
    return 0 if receipt.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
