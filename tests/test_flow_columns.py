"""A frame missing a spec column fails by name, before any tensor op."""

# %% imports ---------------------------------------------------------------------------
import pytest

from tramdag import LS, CausalFlowDAG, ContinuousNode


# %% public functions ------------------------------------------------------------------
def test_missing_column_is_named(ls_chain):
    df = ls_chain["draw"](100, 0)[["x1", "x2"]]
    flow = CausalFlowDAG({"x1": ContinuousNode(), "x2": ContinuousNode([LS("x1")])})
    with pytest.raises(KeyError, match=r"\['x2'\]"):
        flow.fit(df.drop(columns=["x2"]), epochs=1)
    with pytest.raises(KeyError, match=r"\['x1'\]"):
        flow.log_prob(df.rename(columns={"x1": "X1"}))
