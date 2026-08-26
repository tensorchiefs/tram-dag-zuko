"""Helpers that are useful around a fit without being part of one.

:func:`config_section` picks a section out of an already-parsed
configuration. It is not about causal modelling, which is why it lives here
rather than in a modelling module, and it imports nothing: reading a config
needs no dependency at all.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

# %% global variables ------------------------------------------------------------------
__all__ = ["config_section"]


# %% public functions ------------------------------------------------------------------
def config_section(document: dict, *keys: str) -> dict:
    """Pick a mapping out of a parsed configuration.

    Parsing is the caller's job — pass whatever ``yaml.safe_load``,
    ``json.load`` or ``tomllib.load`` returned. This descends through
    ``keys`` and gives the mapping found there as a shallow copy.

    Parameters
    ----------
    document : dict
        The parsed configuration.
    *keys : str
        Keys to descend through before the mapping is returned, for example
        ``"variants", "atan-cs"`` for a document that groups several
        variants. Without any key the document itself is used.

    Returns
    -------
    dict
        The selected mapping, as a shallow copy.

    Raises
    ------
    KeyError
        If one of ``keys`` is not present. The message lists what is
        available at that level.
    ValueError
        If a selected value is not a mapping.

    Examples
    --------
    >>> document = {"variants": {"fast": {"epochs": 5, "lr": 0.01}}}
    >>> config_section(document, "variants", "fast")
    {'epochs': 5, 'lr': 0.01}
    """
    node = document
    for depth, key in enumerate(keys):
        if not isinstance(node, dict):
            # malformed config data, not a wrongly typed argument
            raise ValueError(  # noqa: TRY004
                f"{' -> '.join(keys[:depth]) or 'the document'} is "
                f"{type(node).__name__}, not a mapping"
            )
        if key not in node:
            raise KeyError(
                f"no '{key}' in {' -> '.join(keys[:depth]) or 'the document'}. "
                f"Available: {', '.join(sorted(map(str, node)))}"
            )
        node = node[key]

    where = " -> ".join(keys) or "the document"
    if not isinstance(node, dict):
        # malformed config data, not a wrongly typed argument
        raise ValueError(f"{where} is {type(node).__name__}, not a mapping")  # noqa: TRY004

    return dict(node)
