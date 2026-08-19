"""Tests for tramdag.load_config.

The point of the function is the refusal, not the reading: a missing key must
never become a hidden default, and an extra key must never look effective.
"""

import pytest

from tramdag import load_config

CONFIG = """
shared: &shared
  epochs: 500
  learning_rate: 0.001

variants:
  fast:
    <<: *shared
    epochs: 5
  slow:
    <<: *shared
scalar: 3
"""


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(CONFIG)
    return path


def test_reads_a_variant_and_applies_the_anchor(config_file):
    """YAML's merge key does the sharing, so the code needs no defaults."""
    fast = load_config(config_file, "variants", "fast")
    assert fast == {"epochs": 5, "learning_rate": 0.001}
    assert load_config(config_file, "variants", "slow")["epochs"] == 500


def test_require_accepts_an_exact_match(config_file):
    config = load_config(
        config_file, "variants", "fast", require={"epochs", "learning_rate"}
    )
    assert set(config) == {"epochs", "learning_rate"}


def test_missing_key_is_an_error_not_a_default(config_file):
    with pytest.raises(ValueError, match="missing keys \\['batch_size'\\]"):
        load_config(
            config_file,
            "variants",
            "fast",
            require={"epochs", "learning_rate", "batch_size"},
        )


def test_unknown_key_is_an_error_not_ignored(config_file):
    with pytest.raises(ValueError, match="unknown keys \\['learning_rate'\\]"):
        load_config(config_file, "variants", "fast", require={"epochs"})


def test_unknown_section_names_what_is_available(config_file):
    with pytest.raises(KeyError, match="fast, slow"):
        load_config(config_file, "variants", "medium")


def test_top_level_mapping_without_keys(config_file):
    document = load_config(config_file)
    assert set(document) == {"shared", "variants", "scalar"}


def test_descending_into_a_scalar_is_an_error(config_file):
    with pytest.raises(ValueError, match="not a mapping"):
        load_config(config_file, "scalar")


def test_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such config file"):
        load_config(tmp_path / "absent.yaml")


def test_returns_a_copy(config_file):
    """Mutating the result must not affect a later read of the same file."""
    first = load_config(config_file, "variants", "fast")
    first["epochs"] = 999
    assert load_config(config_file, "variants", "fast")["epochs"] == 5
