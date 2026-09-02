"""The seeded state-dict smoke as a test: the RNG-stream/key-layout tripwire.

`tests/tools/statedict_smoke.py` records four seeded flows' parameters; this
wrapper compares the current build bit-for-bit, so a refactor that reorders
parameter construction (tests pass, pinned experiment centers move) fails
HERE instead of in a replication.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "statedict_smoke", Path(__file__).parent / "tools" / "statedict_smoke.py"
)
_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smoke)
BASELINE, build = _smoke.BASELINE, _smoke.build


def test_seeded_state_dicts_match_the_recorded_baseline():
    import torch

    assert BASELINE.exists(), "baseline missing — run statedict_smoke.py record"
    baseline = torch.load(BASELINE, weights_only=True)
    current = build()
    for name, params in baseline.items():
        assert set(current[name]) == set(params), f"{name}: key layout changed"
        for k, v in params.items():
            assert torch.equal(v, current[name][k]), f"{name}: {k} differs"
