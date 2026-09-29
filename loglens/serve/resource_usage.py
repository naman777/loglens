"""Peak resident memory of this process, cross-platform (Windows via psutil, POSIX via resource)."""
from __future__ import annotations


def peak_rss_mb() -> float:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return float(getattr(info, "peak_wset", info.rss)) / 1e6
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
