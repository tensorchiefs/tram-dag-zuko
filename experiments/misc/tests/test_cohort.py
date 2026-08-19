"""Checks on the frozen cohort behind the classical-MLE validation.

This cohort has **no generator in this repository** — it left with the stroke
storyline — so nothing can regenerate it from a seed. Its schema, size and
committed R reference are therefore the only contract it has, and these tests
are that contract.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

COHORT = Path(__file__).resolve().parents[1] / "data" / "magic-mrclean" / "ls"
COLUMNS = ["Age", "mRS_pre", "NIHSSa", "T", "mRS_3m"]
LEVELS = {"mRS_pre": 6, "T": 2, "mRS_3m": 7}


@pytest.fixture(scope="module")
def truth():
    return json.loads((COHORT / "truth.json").read_text())


def test_observational_schema_and_size(truth):
    observed = pd.read_csv(COHORT / "obs.csv")
    assert list(observed.columns) == COLUMNS
    assert len(observed) == truth["n_obs"]
    for column, levels in LEVELS.items():
        values = observed[column].to_numpy()
        assert values.min() >= 0 and values.max() <= levels - 1
        # every level must be populated, or a one-hot column would be all-zero
        assert len(np.unique(values)) == levels


def test_trial_arm_is_balanced_and_complete(truth):
    trial = pd.read_csv(COHORT / "rct.csv")
    assert list(trial.columns) == COLUMNS
    assert not trial.isna().to_numpy().any()
    # randomized assignment: both arms present in comparable numbers
    share_treated = float(trial["T"].mean())
    assert 0.35 < share_treated < 0.65


def test_truth_records_a_confounded_contrast(truth):
    """The known effect and the naive contrast must differ.

    Otherwise the experiment would have nothing to demonstrate.
    """
    assert truth["true_ate"] == pytest.approx(0.132, abs=0.01)
    assert truth["naive_obs_diff"] > truth["true_ate"] + 0.05


def test_r_reference_is_present_and_parses():
    reference = pd.read_csv(COHORT / "ref_ls" / "coefficients.csv")
    assert {"node", "term", "estimate"} <= set(reference.columns)
    outcome = reference[reference["node"] == "mRS_3m"].set_index("term")["estimate"]
    for term in ("Age", "NIHSSa", "T"):
        assert np.isfinite(outcome[term])
    # the treatment lowers the latent, that is, improves the outcome
    assert outcome["T"] < 0
