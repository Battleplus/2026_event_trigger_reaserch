import json
import shutil
import unittest
from pathlib import Path

from aggregate_gain import aggregate


class AggregateGainTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / "_aggregate_tmp"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, state, seed, gain, **extra):
        path = self.root / state / f"suffix_{seed}" / "reduction.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"status": "PASS", "state_id": state,
                                    "suffix_seed": seed, "G": gain, **extra}), encoding="utf-8")

    def test_suffix_mean_and_30_30_counts(self):
        for index in range(2):
            state = f"s{index}"
            self.write(state, 730201, 1.0 if index == 0 else -1.0)
            self.write(state, 730202, 3.0 if index == 0 else -2.0)
        result = aggregate(self.root, self.root / "aggregate.json")
        self.assertEqual(result["valid_state_count"], 2)
        self.assertEqual(result["positive_count"], 1)
        self.assertEqual(result["nonpositive_count"], 1)
        self.assertAlmostEqual(result["states"][0]["gain_variance"], 2.0)

    def test_incomplete_suffix_is_excluded(self):
        self.write("only-one", 730201, 1.0)
        result = aggregate(self.root, self.root / "aggregate.json")
        self.assertEqual(result["valid_state_count"], 0)
        self.assertEqual(result["states"][0]["unknown_reason"], "requires_two_valid_distinct_suffixes")


if __name__ == "__main__":
    unittest.main()
