"""Small helpers that are useful around a fit without being part of one.

Currently one: :func:`load_config`, a YAML reader that refuses a
configuration whose keys do not match what the caller says it reads. The
point is not the file handling — it is that a *missing* key can never become
a hidden default and an *extra* key can never look effective. Any script that
keeps its hyperparameters in a file rather than in code wants that guarantee,
which is why it lives here rather than being copied into each caller.

PyYAML is imported lazily and declared as the optional ``config`` extra, so
installing ``tramdag`` does not pull it in for users who never call this.
"""

from __future__ import annotations

from pathlib import Path


def _yaml():
    """Import PyYAML, with an error that names the extra to install."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the env
        raise ModuleNotFoundError(
            "load_config needs PyYAML. Install it with "
            "`pip install 'tramdag[config]'` (or add pyyaml to your project)."
        ) from exc
    return yaml


def load_config(path: str | Path, *keys: str, require: set[str] | None = None) -> dict:
    """Read a mapping out of a YAML file, and check its keys exactly.

    Parameters
    ----------
    path : str | Path
        The YAML file.
    *keys : str
        Keys to descend through before the mapping is returned, for example
        ``"variants", "atan-cs"`` for a file that groups several variants.
        Without any key the top-level mapping is used.
    require : set[str] | None, optional
        The keys the caller reads. The mapping must carry exactly these:
        anything missing, and anything extra, is an error. Default ``None``
        skips the check.

    Returns
    -------
    dict
        The selected mapping, as a shallow copy.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If one of ``keys`` is not in the document. The message lists what is
        available at that level.
    ValueError
        If the mapping's keys do not match ``require``, or if the selection
        is not a mapping at all.

    Examples
    --------
    >>> load_config(  # doctest: +SKIP
    ...     Path(__file__).with_suffix(".yaml"),
    ...     "variants",
    ...     "atan-cs",
    ...     require={"epochs", "learning_rate"},
    ... )
    {'epochs': 500, 'learning_rate': 0.001}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such config file: {path}")

    node = _yaml().safe_load(path.read_text())
    for key in keys:
        if not isinstance(node, dict):
            raise ValueError(f"{path.name}: '{key}' is not inside a mapping")
        if key not in node:
            raise KeyError(
                f"{path.name}: no '{key}' here. Available: "
                f"{', '.join(sorted(map(str, node)))}"
            )
        node = node[key]

    if not isinstance(node, dict):
        raise ValueError(
            f"{path.name}: {' -> '.join(keys) or 'the document'} is "
            f"{type(node).__name__}, not a mapping"
        )

    config = dict(node)
    if require is not None:
        missing = set(require) - set(config)
        unknown = set(config) - set(require)
        if missing or unknown:
            where = " -> ".join(keys) or path.name
            raise ValueError(
                f"{path.name}, {where}: missing keys {sorted(missing)}, "
                f"unknown keys {sorted(unknown)}"
            )
    return config
