"""Scoped control of FAISS's process-global OpenMP thread count.

``faiss.omp_set_num_threads`` mutates process-wide state. Calling it without
restoring the previous value permanently pins every subsequent FAISS operation
in the process to that thread count, including ones that deliberately manage
their own threading. These helpers make the change scoped instead.
"""

import platform
from contextlib import contextmanager

import faiss


def _supports_omp_control() -> bool:
    return hasattr(faiss, "omp_set_num_threads") and hasattr(faiss, "omp_get_max_threads")


@contextmanager
def faiss_threads(num_threads: int):
    """Temporarily set FAISS's OpenMP thread count, restoring it on exit."""
    if not _supports_omp_control():
        yield
        return

    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(num_threads)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(previous)


@contextmanager
def faiss_single_thread_on_macos():
    """Pin FAISS to one thread on macOS only, where nested OpenMP is unstable."""
    if platform.system() != "Darwin":
        yield
        return
    with faiss_threads(1):
        yield
