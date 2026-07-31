"""Filesystem primitives for modelctl's state layer.

Two rules every state write follows:

- atomic_write_text(): temp file in the target directory + fsync +
  os.replace. A crash mid-write can never leave a truncated JSON/YAML
  file for every later reader to choke on; readers see the old bytes or
  the new bytes, nothing in between.

- state_lock(): one advisory flock serializing state mutations
  (profiles, defaults, the llama-swap config.yaml merge) across
  processes -- the CLI and the web console's job workers write the same
  files. Reentrant within a thread so composed operations can hold it
  around their whole read-modify-write.

This module must stay a leaf: no modelctl imports (modelctl_paths only).
"""
import fcntl
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

from modelctl_paths import STATE_DIR

_LOCK_NAME = ".state.lock"
_tls = threading.local()


def atomic_write_text(path, text):
    """Write text to path so readers never observe a partial file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@contextmanager
def state_lock():
    """Serialize state mutations across processes and threads.

    flock, not lockf: inherited-fd surprises don't apply (we open per
    acquisition), and two threads of one process correctly contend via
    their separate open file descriptions. Reentrant per thread.
    """
    depth = getattr(_tls, "depth", 0)
    if depth:
        _tls.depth = depth + 1
        try:
            yield
        finally:
            _tls.depth -= 1
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    f = open(STATE_DIR / _LOCK_NAME, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        _tls.depth = 1
        try:
            yield
        finally:
            _tls.depth = 0
            fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        f.close()
