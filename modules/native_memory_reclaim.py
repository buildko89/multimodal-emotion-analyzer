"""Return free native pages after an isolated visual request has fully returned."""
import ctypes
from functools import lru_cache
import gc
import os
import sys
import time


@lru_cache(maxsize=1)
def _load_api():
    if sys.platform!='linux':return None,None
    try:
        libc=ctypes.CDLL(None)
        version=libc.gnu_get_libc_version
        version.argtypes=[];version.restype=ctypes.c_char_p
        trim=libc.malloc_trim
        trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int
        return trim,version().decode('ascii')
    except (AttributeError,OSError):return None,None


def release_idle_pages(*,collect_cycles=False):
    """Operate on already-free pages; unsupported runtimes preserve inference.

    Called only after visual() returns, outside model processing and graph close.
    Optional Python collection is explicit; native-only operation is the default.
    """
    started=time.monotonic_ns()
    try:
        trim,version=_load_api()
        if trim is None:return dict(status='unavailable',elapsed_ns=time.monotonic_ns()-started)
        collected=gc.collect() if collect_cycles else None
        returned=trim(0)
        if returned not in (0,1):raise RuntimeError('Unexpected malloc_trim return value')
        return dict(status='released' if returned else 'no_free_pages',glibc=version,arena_max=os.environ.get('MALLOC_ARENA_MAX'),
            python_collected=collected,elapsed_ns=time.monotonic_ns()-started)
    except Exception as error:
        # Reclamation is a resource optimization; preserve the validated output.
        return dict(status='error',reason=f'{type(error).__name__}: {error}',elapsed_ns=time.monotonic_ns()-started)
