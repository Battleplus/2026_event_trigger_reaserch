from pathlib import Path

from civilian import v4_runner


def test_model_samples_are_read_from_scenario_shards_with_bounded_cache(monkeypatch):
    paths = [Path(f"scene-{shard:02d}.json.gz") for shard in range(10)]
    monkeypatch.setattr(v4_runner, "_read_samples",
                        lambda path: [{"sample_id": f"sample-{int(path.name.split('-')[1].split('.')[0])}"}])
    samples = v4_runner._ModelSampleShards(paths)
    assert len(samples) == 10
    assert samples[5]["sample_id"] == "sample-5"
    for index in range(10):
        assert samples[index]["sample_id"] == f"sample-{index}"
    assert len(samples.cache) == 8
