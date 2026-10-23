import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_remaining_batch import select_scenes


def test_existing_old_scene_directory_is_never_selected(tmp_path):
    rows = [{"run_id": "scene-a", "output_path": "scenes/scene-a"},
            {"run_id": "scene-b", "output_path": "scenes/scene-b"}]
    (tmp_path / "scenes" / "scene-a").mkdir(parents=True)
    selected, references = select_scenes(rows, tmp_path)
    assert [row["run_id"] for row in selected] == ["scene-b"]
    assert references[0]["status"] == "EXISTING_NO_FINAL_RECEIPT"


def test_scene_slice_skips_prior_new_output(tmp_path):
    rows = [{"run_id": "scene-a", "output_path": "scenes/scene-a"},
            {"run_id": "scene-b", "output_path": "scenes/scene-b"},
            {"run_id": "scene-c", "output_path": "scenes/scene-c"}]
    prior = tmp_path / "prior"
    (prior / "scenes" / "scene-b").mkdir(parents=True)
    selected, references = select_scenes(rows, tmp_path / "old", [prior], {"scene-a", "scene-b"})
    assert [row["run_id"] for row in selected] == ["scene-a"]
    assert references[0]["run_id"] == "scene-b"
    assert references[0]["source"] == "prior_new_output"
