import json
from pathlib import Path

import pytest

from repurpose.academic.state import AcademicState


def prepare(tmp_path):
    state = AcademicState(tmp_path)
    state.initialize()
    artifact = tmp_path / "proof.txt"
    artifact.write_text("verified artifact")
    state.start("A00", "test start")
    state.finish("A00", [artifact], "complete")
    return state, artifact


def test_artifact_hash_and_dependency_invalidation(tmp_path):
    state, artifact = prepare(tmp_path)
    assert state.verified("A00")
    state.start("A01", "audit")
    state.finish("A01", [artifact], "audit finished")
    assert state.verified("A01")
    artifact.write_text("changed after validation")
    assert not state.verified("A00") and not state.verified("A01")
    with pytest.raises(ValueError, match="Önkoşul"):
        state.start("A02", "should not start")


def test_interrupted_attempt_is_preserved_and_resumable(tmp_path):
    state, artifact = prepare(tmp_path)
    first = state.start("A01", "first audit")
    new_state = AcademicState(tmp_path)
    assert new_state.read()["stages"]["A01"]["status"] == "in_progress"
    second = new_state.start("A01", "resumed audit")
    assert first != second
    records = new_state.read()["stages"]["A01"]["attempts"]
    assert records[0]["status"] == "interrupted"
    new_state.finish("A01", [artifact], "done")
    assert new_state.verified("A01")
    events = [json.loads(line) for line in new_state.journal.read_text().splitlines()]
    assert len(events) >= 6


def test_failed_and_missing_artifact_never_validate(tmp_path):
    state, _ = prepare(tmp_path)
    state.start("A01", "audit")
    with pytest.raises(FileNotFoundError):
        state.finish("A01", [Path("missing.txt")], "done")
    assert not state.verified("A01")
    state.fail("A01", "missing input")
    assert state.read()["stages"]["A01"]["status"] == "failed"
    assert "missing input" in (tmp_path / "ACADEMIC_PROGRESS.md").read_text()


def test_path_outside_project_is_not_evidence(tmp_path):
    state, _ = prepare(tmp_path)
    outside = tmp_path.parent / "outside-proof.txt"
    outside.write_text("not in project")
    state.start("A01", "audit")
    with pytest.raises(ValueError):
        state.finish("A01", [outside], "done")
    assert not state.verified("A01")


def test_deleted_evidence_invalidates_validated_stage(tmp_path):
    state, artifact = prepare(tmp_path)
    artifact.unlink()
    assert not state.verified("A00")


def test_finish_refuses_dependency_changed_during_work(tmp_path):
    state, artifact = prepare(tmp_path)
    state.start("A01", "audit started")
    artifact.write_text("dependency changed while work was running")
    with pytest.raises(ValueError, match="çalışma sırasında değişti"):
        state.finish("A01", [artifact], "must not validate")
    assert not state.verified("A01")
