"""Every YAML key is read by its script; the loader no longer checks key sets.

A misspelled or leftover key would otherwise be silent, so this test keeps the
config files and the code that reads them from drifting apart.
"""

# %% imports ---------------------------------------------------------------------------
import re
from pathlib import Path

import pytest
import yaml

# %% global variables ------------------------------------------------------------------
EXPERIMENTS = Path(__file__).resolve().parents[2]
CONFIGS = sorted(EXPERIMENTS.glob("*/*.yaml"))  # the same glob the workflow plans from


# %% private functions -----------------------------------------------------------------
def _keys_read(script: Path, pattern: str) -> set[str]:
    """Keys the script reads by string subscript, its area helpers included."""
    source = script.read_text()
    for helper in ("helpers.py", "common.py"):
        sibling = script.parent / helper
        if sibling.exists():
            source += sibling.read_text()
    return set(re.findall(pattern, source))


# %% public functions ------------------------------------------------------------------
@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_yaml_key_is_read_by_the_script(config):
    document = yaml.safe_load(config.read_text())
    if "variants" in document:
        # variant configs are read through the one `config` dict
        sections = document["variants"]
        read = _keys_read(
            config.with_suffix(".py"), r"""config\[["']([a-z_0-9]+)["']\]"""
        )
    else:
        # workload-shaped configs (bench_training) are read through several
        # local names, so any string subscript counts as read
        sections = {**document["workloads"], "lbfgs": document["lbfgs"]}
        read = _keys_read(config.with_suffix(".py"), r"""\[["']([a-z_0-9]+)["']\]""")
    for section, values in sections.items():
        unread = set(values) - read
        assert not unread, (
            f"{config.name}[{section}]: keys nothing reads: {sorted(unread)}"
        )
