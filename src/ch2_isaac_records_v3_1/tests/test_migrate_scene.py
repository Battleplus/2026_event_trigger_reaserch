import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from migrate_scene import migrate_scene


def test_missing_public_capture_is_explicit(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    result = migrate_scene(source, tmp_path / "new")
    assert result["status"] == "conversion_unavailable"
    manifest = json.loads((tmp_path / "new" / "manifest.json").read_text())
    assert manifest["status"] == "conversion_unavailable"
    assert not (tmp_path / "new" / "records.jsonl").exists()


def test_legacy_public_capture_is_converted(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    static = {"map_size_km": 300, "roi_width_km": 50,
              "public_task_geometry": {}}
    (source / "public_capture").mkdir()
    capture = source / "public_capture"
    (capture / "static_config.json").write_text(json.dumps(static))
    digest = hashlib.sha256(json.dumps(static, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    row = {"sim_time_s": 0, "static_config_sha256": digest,
           "deltas": {"aircraft": {"upsert": {}, "delete": []},
                       "public_tasks": {"upsert": {}, "delete": []},
                       "observations": {"upsert": {}, "delete": []}},
           "events": [], "decisions": [], "costs": []}
    with gzip.open(capture / "ticks.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    result = migrate_scene(source, tmp_path / "new")
    assert result["status"] == "converted_legacy"
    assert (tmp_path / "new" / "config.json").is_file()
    assert (tmp_path / "new" / "records.jsonl").is_file()
    assert (tmp_path / "new" / "labels" / "state_index.json").is_file()
    manifest = json.loads((tmp_path / "new" / "manifest.json").read_text())
    assert manifest["public_records_sha256"] == hashlib.sha256(
        (tmp_path / "new" / "records.jsonl").read_bytes()).hexdigest()
