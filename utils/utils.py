from contextlib import contextmanager
from timeit import default_timer
import torch
from .data import pad

@contextmanager
def elapsed_timer():
    """Context manager to easily time executions.
    
    Usage:
    >>> with elapsed_timer() as t:
    ...     pass # some instructions
    >>> elapsed_time = t()
    """
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
