"""Tests for tramdag.utils.

`config_section` exists for its refusals: a missing key must never become a
hidden default, and an extra key must never look effective. Parsing is the
caller's job, so these tests hand it plain dicts.
"""

# %% imports ---------------------------------------------------------------------------
import pathlib

import pytest

import tramdag as td
from tramdag.utils import config_section, machine_info

# %% global variables ------------------------------------------------------------------
DOCUMENT = {
    "variants": {
        "fast": {"epochs": 5, "learning_rate": 0.001},
        "slow": {"epochs": 500, "learning_rate": 0.001},
    },
    "scalar": 3,
}


# %% public functions ------------------------------------------------------------------
def test_selects_a_nested_section():
    assert config_section(DOCUMENT, "variants", "fast") == {
        "epochs": 5,
        "learning_rate": 0.001,
    }


def test_no_keys_gives_the_document():
    assert set(config_section(DOCUMENT)) == {"variants", "scalar"}


def test_unknown_section_names_what_is_available():
    with pytest.raises(KeyError, match="fast, slow"):
        config_section(DOCUMENT, "variants", "medium")


def test_descending_into_a_scalar_is_an_error():
    with pytest.raises(ValueError, match="not a mapping"):
        config_section(DOCUMENT, "scalar", "deeper")


def test_selecting_a_scalar_is_an_error():
    with pytest.raises(ValueError, match="not a mapping"):
        config_section(DOCUMENT, "scalar")


def test_returns_a_copy():
    """Mutating the result must not touch the caller's document."""
    section = config_section(DOCUMENT, "variants", "fast")
    section["epochs"] = 999
    assert DOCUMENT["variants"]["fast"]["epochs"] == 5


def test_the_package_needs_no_yaml_parser():
    """The helper takes parsed data, so installing tramdag pulls in no parser.

    Checked against the declared dependencies, which is the actual contract a
    user gets; the experiments keep pyyaml in their own dependency group.
    """
    tomllib = pytest.importorskip(
        "tomllib", reason="python 3.10 has no tomllib; the check runs on 3.11+"
    )

    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():  # installed without the sources
        pytest.skip("no pyproject.toml next to the tests")
    project = tomllib.loads(pyproject.read_text())["project"]
    assert not [d for d in project["dependencies"] if "yaml" in d.lower()]


# %% machine_info ----------------------------------------------------------------------
def test_machine_info_has_expected_fields():
    assert td.machine_info is machine_info  # the package re-export
    info = machine_info()
    for key in ("hostname", "os", "python", "torch", "tramdag", "cpu_count"):
        assert key in info
    assert info["torch"]
    assert info["python"]
