"""Tests for the API papercuts in issue #12: constructor seeding, history +
machine-info persistence through save/load, and the machine_info() helper.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import tramdag as td
from tramdag import LS, CausalFlowDAG, ContinuousNode, OrdinalNode


def _spec():
    return {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([LS("x1")]),
        "y": OrdinalNode(3, [LS("x1")]),
    }


# -------------------------------------------------- #1 constructor seeding
def test_constructor_seed_makes_init_reproducible():
    a = CausalFlowDAG(_spec(), seed=42)
    b = CausalFlowDAG(_spec(), seed=42)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_constructor_seed_differs_across_seeds():
    a = CausalFlowDAG(_spec(), seed=42)
    c = CausalFlowDAG(_spec(), seed=7)
    assert any(
        not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters())
    )


def test_no_seed_still_works():
    # default (no seed) constructs fine; just not pinned across processes
    flow = CausalFlowDAG(_spec())
    assert flow.order  # built


# ------------------------------------------- #2 history persists through io
def test_save_load_round_trips_history(tmp_path):
    df = pd.DataFrame(
        {
            "x1": np.random.randn(200),
            "x2": np.random.randn(200),
            "y": np.random.randint(0, 3, 200).astype(float),
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
def test_machine_info_has_expected_fields():
    info = td.machine_info()
    for k in [
        "hostname",
        "os",
        "cpu_count",
        "python",
        "torch",
        "zuko",
        "tramdag",
        "cuda",
        "mps",
        "ram_gb",
    ]:
        assert k in info
    assert info["torch"] == torch.__version__
    assert info["tramdag"] == td.__version__


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
