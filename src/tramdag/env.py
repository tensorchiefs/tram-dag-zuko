"""Machine / environment provenance — captured with saved models so cached
timing and benchmark numbers are interpretable across machines.
"""

from __future__ import annotations

import os
import platform
import socket

import torch

__all__ = ["machine_info"]


def machine_info() -> dict:
    """A best-effort snapshot of the machine and software environment:
    host, OS, CPU/GPU, cores, RAM, and the python/torch/zuko/tramdag versions.
    Fails open — any field that can't be read is ``None`` rather than raising.
    """
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
    try:
        import zuko

        info["zuko"] = zuko.__version__
    except Exception:
        info["zuko"] = None
    try:
        from . import __version__

        info["tramdag"] = __version__
    except Exception:
        info["tramdag"] = None
    try:  # total RAM (POSIX)
        info["ram_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1
        )
    except (ValueError, OSError, AttributeError):
        info["ram_gb"] = None
    return info
