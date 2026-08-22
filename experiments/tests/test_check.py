"""Tests for the ground-truth comparison every experiment job runs.

``check.py`` is the gate on all ten CI jobs and is shared by every area, so it
lives beside them rather than inside one. It had no test until an audit pointed
out that its band-and-``why`` logic had never executed on a real input — every
committed bound was inside the band, so the branch was dead in practice.

Each test writes a throwaway metrics/ground-truth pair into ``tmp_path`` and
points the module's ``HERE`` at it, so nothing here reads or writes a committed
file.
"""

from __future__ import annotations

import json

import check as check_module
import pytest


@pytest.fixture
def area(tmp_path, monkeypatch):
    """Give a writer for one throwaway area, with ``check`` pointed at it."""
    monkeypatch.setattr(check_module, "HERE", tmp_path)

    def write(metrics: dict, truth: dict, name: str = "toy") -> str:
        results = tmp_path / "paper" / "results" / name
        results.mkdir(parents=True, exist_ok=True)
        (results / "metrics.json").write_text(json.dumps(metrics))
        gt = tmp_path / "paper" / "ground_truth"
        gt.mkdir(parents=True, exist_ok=True)
        (gt / f"{name}.json").write_text(json.dumps(truth))
        return name

    return write


def test_value_entry_passes_inside_atol_and_fails_outside(area):
    name = area({"beta": 1.02}, {"beta": {"value": 1.0, "atol": 0.05}})
    failures, notes, unchecked, _loose = check_module.compare("paper", name)
    assert not failures
    assert len(notes) == 1
    assert not unchecked

    name = area({"beta": 1.2}, {"beta": {"value": 1.0, "atol": 0.05}}, "toy2")
    failures, _, _, _ = check_module.compare("paper", name)
    assert len(failures) == 1
    assert "deviation" in failures[0]


def test_max_entry_only_fails_upward(area):
    """A better fit must not fail the run — that is the whole point of {max}."""
    truth = {"err": {"max": 0.2}}
    name = area({"err": 0.11}, truth)
    failures, _, _, _ = check_module.compare("paper", name)
    assert not failures  # 0.11 is well under the bound

    name = area({"err": 0.0001}, truth, "tiny")
    failures, _, _, _ = check_module.compare("paper", name)
    assert not failures  # far better than the bound is still a pass

    name = area({"err": 0.3}, truth, "over")
    failures, _, _, _ = check_module.compare("paper", name)
    assert len(failures) == 1
    assert "exceeds its bound" in failures[0]


def test_a_bound_far_above_its_measurement_is_reported(area):
    name = area({"err": 0.01}, {"err": {"max": 0.2}})  # 20x
    _, _, _, loose = check_module.compare("paper", name)
    assert len(loose) == 1
    assert "too loose" in loose[0]


def test_a_bound_hugging_its_measurement_is_reported(area):
    name = area({"err": 0.19}, {"err": {"max": 0.2}})  # 1.05x
    _, _, _, loose = check_module.compare("paper", name)
    assert len(loose) == 1
    assert "another machine" in loose[0]


def test_why_excuses_a_wide_bound_but_never_a_tight_one(area):
    """The asymmetry matters: no argument survives a bound below 1.5x."""
    name = area({"err": 0.01}, {"err": {"max": 0.2, "why": "measured elsewhere"}})
    _, notes, _, loose = check_module.compare("paper", name)
    assert not loose
    assert "deliberately wide" in notes[0]

    name = area({"err": 0.19}, {"err": {"max": 0.2, "why": "measured"}}, "tight")
    _, _, _, loose = check_module.compare("paper", name)
    assert len(loose) == 1
    assert "another machine" in loose[0]


def test_a_center_drifting_through_its_tolerance_is_reported(area):
    """The failure mode that let a stale center pass: 62% of atol consumed."""
    name = area({"beta": 1.031}, {"beta": {"value": 1.0, "atol": 0.05}})
    failures, _, _, loose = check_module.compare("paper", name)
    assert not failures
    assert len(loose) == 1
    assert "older run" in loose[0]

    name = area({"beta": 1.01}, {"beta": {"value": 1.0, "atol": 0.05}}, "fresh")
    _, _, _, loose = check_module.compare("paper", name)
    assert not loose


def test_a_truth_entry_the_run_stopped_producing_is_an_error(area):
    name = area({"beta": 1.0}, {"gone": {"value": 1.0, "atol": 0.05}})
    failures, _, _, _ = check_module.compare("paper", name)
    assert len(failures) == 1
    assert "no such metric" in failures[0]


def test_a_metric_without_an_entry_is_reported_not_failed(area):
    name = area({"beta": 1.0, "extra": 7.0}, {"beta": {"value": 1.0, "atol": 0.05}})
    failures, _, unchecked, _ = check_module.compare("paper", name)
    assert not failures
    assert len(unchecked) == 1
    assert "extra" in unchecked[0]


def test_underscore_keys_are_notes_for_the_reader(area):
    name = area({"beta": 1.0}, {"_note": "prose", "beta": {"value": 1.0, "atol": 0.05}})
    failures, notes, _, _ = check_module.compare("paper", name)
    assert not failures
    assert len(notes) == 1


def test_a_missing_ground_truth_file_says_what_to_do(area, tmp_path):
    """A run with no committed expectations names the next step, not a KeyError."""
    results = tmp_path / "paper" / "results" / "never-pinned"
    results.mkdir(parents=True)
    (results / "metrics.json").write_text('{"beta": 1.0}')
    with pytest.raises(FileNotFoundError, match="no ground truth"):
        check_module.compare("paper", "never-pinned")


def test_a_missing_metrics_file_says_so(area):
    area({"beta": 1.0}, {"beta": {"value": 1.0, "atol": 0.05}})
    with pytest.raises(FileNotFoundError, match="no metrics to check"):
        check_module.compare("paper", "never-run")
