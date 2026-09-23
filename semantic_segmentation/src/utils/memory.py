from __future__ import annotations

import ctypes
import gc
import logging
import os
import platform


def log_memory_usage(
    stage: str,
    enabled: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    if not enabled:
        return
    logger = logger or logging.getLogger(__name__)
    try:
        import psutil
    except ImportError:
        logger.debug("psutil is unavailable; RAM logging disabled")
        return

    process = psutil.Process(os.getpid())
    main_rss = process.memory_info().rss
    children = [
        child for child in process.children(recursive=True) if child.is_running()
    ]
    children_rss = 0
    for child in children:
        try:
            children_rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    gib = 1024**3
    logger.info(
        "RAM | stage=%s main=%.2f GiB workers=%.2f GiB children=%d "
        "available=%.2f GiB",
        stage,
        main_rss / gib,
        children_rss / gib,
        len(children),
        psutil.virtual_memory().available / gib,
    )


def release_host_memory() -> None:
    gc.collect()
    if platform.system() == "Linux":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
