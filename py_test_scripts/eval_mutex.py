"""
Global lock variable and initialization
"""
from NamedAtomicLock import NamedAtomicLock
import warnings


def init_lock(name='ev_lock'):
    global lock
    lock = NamedAtomicLock(name)


def acquire_lock():
    global acquired
    if lock.acquire(timeout=5):
        acquired = True
    else:
        warnings.warn('Writing to yml file takes too long. The next operation might corrupt a file.', UserWarning)
        acquired = False


def release_lock():
    if acquired:
        if not lock.release():
            warnings.warn('Unable to release Lock.', UserWarning)