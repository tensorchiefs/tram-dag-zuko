"""Helpers that are useful around a fit without being part of one.

Two of them: :func:`config_section`, which picks a section out of an
already-parsed configuration, and :func:`machine_info`, the environment
snapshot ``save`` stores with a model.

Neither is about causal modelling, which is why they live together here
rather than being spread through the modelling modules.

Nothing is imported at module level: reading a config needs no dependency at
all, and the environment snapshot pulls in ``torch`` and ``platform`` only
when it is actually called.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

# %% global variables ------------------------------------------------------------------
__all__ = ["config_section", "machine_info"]


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


def machine_info() -> dict:
    """Describe the machine and the software environment.

    The snapshot holds the host name, the operating system, the CPU and GPU,
    the core count, the RAM size, and the versions of python, torch, zuko and
    tramdag. ``save`` stores it with the model, so timing and benchmark
    numbers stay comparable across machines.

    Returns
    -------
    dict
        One key per property. ``ram_gb`` is ``None`` off POSIX, where the
        page-count call does not exist.
    """
    import os
    import platform
    import socket

    import torch

    info: dict = {
        "hostname": socket.gethostname().split(".")[0],
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "mps": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    }
    import zuko  # a hard dependency: tramdag cannot import without it

    from . import __version__

    info["zuko"] = zuko.__version__
    info["tramdag"] = __version__
    try:  # total RAM (POSIX)
        info["ram_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1
        )
    except (ValueError, OSError, AttributeError):
        info["ram_gb"] = None
    return info
