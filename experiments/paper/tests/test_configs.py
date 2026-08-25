"""Every YAML key is read by its script; the loader no longer checks key sets.

A misspelled or leftover key would otherwise be silent, so this test keeps the
config files and the code that reads them from drifting apart.
"""

import re
from pathlib import Path

import pytest
import yaml

PAPER = Path(__file__).resolve().parents[1]
SCRIPTS = ["triangle", "triangle_mixed", "vaca", "carefl"]


def _keys_read(script: str) -> set[str]:
    """Keys the script or helpers.py read: ``config["key"]`` and ``*_KEYS`` tuples."""
    source = (PAPER / f"{script}.py").read_text() + (PAPER / "helpers.py").read_text()
    direct = set(re.findall(r'config\["([a-z_0-9]+)"\]', source))
    listed = {
        key
        for block in re.findall(r"_KEYS\s*=\s*[\(\[]([^\)\]]*)[\)\]]", source)
        for key in re.findall(r'"([a-z_0-9]+)"', block)
    }
    return direct | listed


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_yaml_key_is_read_by_the_script(script):
    document = yaml.safe_load((PAPER / f"{script}.yaml").read_text())
    read = _keys_read(script)
    for variant, config in document["variants"].items():
        unread = set(config) - read
        assert not unread, (
            f"{script}.yaml[{variant}]: keys nothing reads: {sorted(unread)}"
        )
