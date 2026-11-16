import json

from mosaic.cli.run_metadata import (
    RUN_METADATA_FILE_NAME,
    collect_run_metadata,
    mark_finished,
    write_run_metadata,
)


def test_metadata_records_commit_hardware_and_completion(tmp_path):
    metadata = collect_run_metadata(
        threads=64,
        gpus_per_sim=0.125,
        simulation_overrides=["experiment_name=test"],
        planner_config={"parameters": {"ablation": "none"}},
    )
    write_run_metadata(metadata, tmp_path)
    written = json.loads((tmp_path / RUN_METADATA_FILE_NAME).read_text())

    assert len(written["git"]["commit"]) == 40
    assert written["hardware"]["threads"] == 64
    assert written["hardware"]["gpus_per_sim"] == 0.125
    assert written["finished_at"] is None

    mark_finished(metadata, tmp_path)
    written = json.loads((tmp_path / RUN_METADATA_FILE_NAME).read_text())
    assert written["finished_at"] is not None
