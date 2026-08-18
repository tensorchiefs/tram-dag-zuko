"""Record which machine produced a result.

``save`` stores this snapshot with the model. Timing and benchmark numbers
are then comparable across machines.
"""

from __future__ import annotations

import os
import platform
import socket

import torch

__all__ = ["machine_info"]


def machine_info() -> dict:
    """Describe the machine and the software environment.

    The snapshot holds the host name, the operating system, the CPU and GPU,
    the core count, the RAM size, and the versions of python, torch, zuko and
    tramdag.

    Returns
    -------
    dict
        One key per property. A property that cannot be read is ``None``.
        This function never raises.
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
