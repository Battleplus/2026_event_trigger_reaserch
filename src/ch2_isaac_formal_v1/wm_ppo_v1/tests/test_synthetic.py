import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wm_ppo_v1.contracts import OneStepTarget, PublicHistoryFrame
from wm_ppo_v1.features import EVENT_DIM, FEATURE_DIM, collate_graph_sequences, public_history_to_graph
from wm_ppo_v1.policies import CostLedger, SixMethod, incremental_cost, method_specs
from wm_ppo_v1.ppo_entry import not_run_receipt


class SyntheticInterfaceTests(unittest.TestCase):
    def _frame(self, n=2):
        return PublicHistoryFrame(
            time_s=1.0,
            platforms=tuple({"x_norm": i / 10, "energy_fraction": 0.8} for i in range(n)),
            tasks=({"priority_norm": 0.5, "active": True},),
            observations=(),
        )

    def test_dynamic_graph_masks_variable_nodes(self):
        graph = public_history_to_graph(self._frame(3))
        self.assertEqual(graph.shape, (4, FEATURE_DIM))
        batch = collate_graph_sequences([[self._frame(3)], [self._frame(1), self._frame(2)]], [
            OneStepTarget({"task_progress": 1, "energy_delta": -0.1, "delay": 0, "risk": 0.2}, 0.3),
            OneStepTarget({"task_progress": 0, "energy_delta": -0.2, "delay": 1, "risk": 0.4}, -0.1),
        ])
        self.assertEqual(batch.node_features.shape[:3], (2, 2, 4))
        self.assertEqual(batch.node_mask.dtype, np.bool_)
        self.assertTrue(np.isnan(batch.gain_target).sum() == 0)

    def test_cost_uses_frozen_terms(self):
        ledger = CostLedger(messages_increment=100, switches_increment=2, planner_calls_increment=1, changed_actors=4, planning_cpu_seconds_increment=3)
        expected = 0.01 * .1 + 0.05 * .02 + 0.05 * .01 + 0.01 * (4 / 36) + 0.01 * 3
        self.assertAlmostEqual(incremental_cost(ledger), expected)

    def test_six_methods_are_fixed(self):
        self.assertEqual(tuple(spec.name for spec in method_specs()), tuple(SixMethod))

    def test_ppo_is_explicitly_not_run(self):
        receipt = not_run_receipt()
        self.assertEqual(receipt.status, "NOT_RUN")
        self.assertEqual(receipt.updates, 0)
        self.assertGreaterEqual(len(receipt.reasons), 2)

    def test_checkpoint_restore_cpu(self):
        try:
            import torch
            from wm_ppo_v1.world_model import EventAwareWorldModel
            from wm_ppo_v1.checkpoint import restore_and_compare, save_checkpoint
        except ImportError:
            self.skipTest("torch unavailable")
        torch.manual_seed(7)
        model = EventAwareWorldModel(FEATURE_DIM, hidden_dim=8)
        x = torch.zeros(1, 2, 3, FEATURE_DIM)
        m = torch.ones(1, 2, 3, dtype=torch.bool)
        s = torch.ones(1, 2, dtype=torch.bool)
        path = Path(__file__).with_name("._synthetic_checkpoint.pt")
        try:
            save_checkpoint(path, model, {"synthetic": True, "one_step": True})
            receipt = restore_and_compare(path, lambda: EventAwareWorldModel(FEATURE_DIM, hidden_dim=8), (x, m, s))
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(receipt.status, "PASS")
        self.assertTrue(receipt.predictions_equal)

    def test_checkpoint_restores_model_optimizer_and_rng_in_fresh_process(self):
        try:
            import torch
            from wm_ppo_v1.world_model import EventAwareWorldModel
            from wm_ppo_v1.checkpoint import save_training_checkpoint, verify_training_checkpoint_in_fresh_process
        except ImportError:
            self.skipTest("torch unavailable")
        torch.manual_seed(17)
        model = EventAwareWorldModel(FEATURE_DIM, hidden_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for parameter in model.parameters():
            optimizer.state[parameter] = {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.ones_like(parameter),
            }
        inputs = (
            torch.zeros(1, 2, 3, FEATURE_DIM),
            torch.ones(1, 2, 3, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
        )
        metadata = {
            "seed": 8201, "dataset_sha256": "a" * 64,
            "feature_names": ["synthetic"], "event_names": ["synthetic"],
            "event_schema_version": 1,
            "feature_dim": FEATURE_DIM, "event_dim": EVENT_DIM,
            "architecture": {"node_dim": FEATURE_DIM, "hidden_dim": 8,
                             "dynamic_dim": 4, "event_dim": EVENT_DIM, "action_dim": 5},
            "optimizer": {"name": "Adam", "lr": 1e-3, "weight_decay": 0.0},
            "normalization": {"scope": "train_split_only", "center": [0.0], "scale": [1.0]},
            "device": {"device": "cpu"},
        }
        path = Path(__file__).with_name("._fresh_process_checkpoint.pt")
        try:
            save_training_checkpoint(path, model, optimizer, metadata, inputs, device="cpu")
            receipt = verify_training_checkpoint_in_fresh_process(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(receipt["status"], "PASS", receipt)
        self.assertTrue(receipt["predictions_equal"])
        self.assertTrue(receipt["optimizer_equal"])

    def test_failed_terminal_gate_writes_receipt_without_device_or_training(self):
        from wm_ppo_v1 import trainer
        failed_dataset = {"gate": {"ready": False, "reason": "BATCH_NOT_TERMINAL"}, "rows": []}
        tests_dir = Path(__file__).parent
        saved_path = tests_dir / "training_receipt.json"
        try:
            with patch("light_models_v1.dataset.load_feature_dataset", return_value=failed_dataset), \
                 patch.object(trainer, "_device_or_reason", side_effect=AssertionError("device must not be checked")), \
                 patch.object(trainer, "_dynamic_step", side_effect=AssertionError("training must not run")):
                receipt = trainer.run_training(tests_dir / "nonexistent_batch", tests_dir)
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
        finally:
            saved_path.unlink(missing_ok=True)
        self.assertEqual(receipt["status"], "NOT_RUN")
        self.assertEqual(saved["reason"], "BATCH_NOT_TERMINAL")
        self.assertEqual(saved["updates"], 0)


if __name__ == "__main__":
    unittest.main()
