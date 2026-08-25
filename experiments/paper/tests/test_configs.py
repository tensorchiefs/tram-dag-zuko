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
PAPER = Path(__file__).resolve().parents[1]
SCRIPTS = ["triangle", "triangle_mixed", "vaca", "carefl"]


# %% private functions -----------------------------------------------------------------
def _keys_read(script: str) -> set[str]:
    """Keys the script or helpers.py read as ``config["key"]``."""
    source = (PAPER / f"{script}.py").read_text() + (PAPER / "helpers.py").read_text()
    return set(re.findall(r'config\["([a-z_0-9]+)"\]', source))


# %% public functions ------------------------------------------------------------------
@pytest.mark.parametrize("script", SCRIPTS)
def test_every_yaml_key_is_read_by_the_script(script):
    document = yaml.safe_load((PAPER / f"{script}.yaml").read_text())
    read = _keys_read(script)
    for variant, config in document["variants"].items():
        unread = set(config) - read
        assert not unread, (
            f"{script}.yaml[{variant}]: keys nothing reads: {sorted(unread)}"
        )
