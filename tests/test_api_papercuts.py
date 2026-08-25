"""Tests for the API papercuts in issue #12: constructor seeding, history +
and machine-info persistence through save/load (the helper itself is
tested in test_utils.py).
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
import torch

import tramdag as td
from tramdag import LS, CausalFlowDAG, ContinuousNode, OrdinalNode


# %% private functions -----------------------------------------------------------------
def _spec():
    return {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([LS("x1")]),
        "y": OrdinalNode(3, [LS("x1")]),
    }


# %% public functions ------------------------------------------------------------------
# -------------------------------------------------- #1 constructor seeding
def test_constructor_seed_makes_init_reproducible():
    a = CausalFlowDAG(_spec(), seed=42)
    b = CausalFlowDAG(_spec(), seed=42)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_constructor_seed_differs_across_seeds():
    a = CausalFlowDAG(_spec(), seed=42)
    c = CausalFlowDAG(_spec(), seed=7)
    assert any(
        not torch.equal(pa, pc)
        for pa, pc in zip(a.parameters(), c.parameters(), strict=True)
    )


def test_no_seed_still_works():
    # default (no seed) constructs fine, and is deliberately not pinned:
    # two unseeded models differ, which is what makes seed= the only knob
    a, b = CausalFlowDAG(_spec()), CausalFlowDAG(_spec())
    assert any(
        not torch.equal(pa, pb)
        for pa, pb in zip(a.parameters(), b.parameters(), strict=True)
    )


# ------------------------------------------- #2 history persists through io
def test_save_load_round_trips_history(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(200),
            "x2": rng.standard_normal(200),
            "y": rng.integers(0, 3, 200).astype(float),
        }
    )
    flow = CausalFlowDAG(_spec(), seed=0)
    flow.fit(df, df, epochs=12, verbose=0)
    assert len(flow.history["val"]) == 12
    p = tmp_path / "flow.pt"
    flow.save(p)
    loaded = CausalFlowDAG.load(p)
    assert set(loaded.history) == {"train", "val", "lr", "time"}
    assert len(loaded.history["val"]) == 12
    assert len(loaded.history["time"]) == 12  # wall-clock curve survives too


# ----------------------------------------- #4 machine/env info in metadata
def test_save_carries_machine_and_version_metadata(tmp_path):
    flow = CausalFlowDAG(_spec(), seed=0)
    p = tmp_path / "flow.pt"
    flow.save(p)
    loaded = CausalFlowDAG.load(p)
    assert {"tramdag_version", "saved_at", "device", "machine"} <= set(loaded.meta)
    assert loaded.meta["tramdag_version"] == td.__version__
    assert loaded.meta["device"] == "cpu"
    assert loaded.meta["machine"]["torch"] == torch.__version__


def test_load_requires_a_complete_checkpoint(tmp_path):
    # save() always writes spec, weights, history and meta; a dict missing
    # any of them is not a tramdag checkpoint and must fail loudly
    flow = CausalFlowDAG(_spec(), seed=0)
    from tramdag.spec import spec_to_dict

    p = tmp_path / "partial.pt"
    torch.save({"spec": spec_to_dict(flow.spec), "state_dict": flow.state_dict()}, p)
    with pytest.raises(KeyError):
        CausalFlowDAG.load(p)


def test_ls_coefficients_skips_network_shifts():
    """A node mixing LS and CS terms gives only its linear-shift weights.

    Reading `.weight` off every shift module used to raise an
    AttributeError on a ComplexShift, which broke the paper's headline
    complex-shift replication (`experiments/paper/triangle.py atan-cs`).
    """
    from tramdag import CS, LS, VC, CausalFlowDAG, ContinuousNode, OrdinalNode

    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([LS("x1")]),
        "t": OrdinalNode(2, [LS("x1")]),
        "x3": ContinuousNode([LS("x1"), CS("x2"), VC("x2", t="t")]),
    }
    coefficients = CausalFlowDAG(spec, seed=0).ls_coefficients()
    assert set(coefficients["x3"]) == {"x1"}  # CS and VC carry no weight
    assert set(coefficients["x2"]) == {"x1"}
    assert coefficients["x3"]["x1"].shape == (1,)


def test_ls_coefficients_omits_a_node_without_linear_shifts():
    from tramdag import CS, CausalFlowDAG, ContinuousNode

    spec = {"x1": ContinuousNode(), "x2": ContinuousNode([CS("x1")])}
    assert CausalFlowDAG(spec, seed=0).ls_coefficients() == {}


def test_fit_rejects_a_batch_size_below_one():
    """batch_size=0 used to reach range() and fail with a cryptic message."""
    df = pd.DataFrame({"x1": np.zeros(8), "x2": np.zeros(8), "y": np.zeros(8)})
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        CausalFlowDAG(_spec(), seed=0).fit(df, epochs=1, batch_size=0)
