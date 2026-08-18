"""Fixtures shared by the test modules.

Only helpers that are provably identical across files live here. Each
module keeps its own DGP builders and specs: those pin the one SCM or
syntax variant a property needs, and sharing them would couple unrelated
acceptance bars.
"""

import pytest
import torch

from tramdag import CausalFlowDAG, ContinuousNode


@pytest.fixture
def fit_x3_nll():
    """Fit ``x1 -> x3 <- x2`` with the given terms; give x3's validation NLL.

    The budget (300 epochs at lr 1e-2, batch 512) is shared by the joint-vs-
    additive and additive-intercept comparisons, which only read the NLL
    difference between two fits of the same shape.
    """

    def _fit(terms, train, val) -> float:
        torch.manual_seed(0)
        flow = CausalFlowDAG(
            {
                "x1": ContinuousNode(),
                "x2": ContinuousNode(),
                "x3": ContinuousNode(terms),
            },
            seed=0,
        )
        flow.fit(train, val, epochs=300, learning_rate=1e-2, batch_size=512, verbose=0)
        return flow.nll(val)["x3"]

    return _fit
