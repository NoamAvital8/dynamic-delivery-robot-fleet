"""Atomic artifacts, input identity, conservative process/RAM admission."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile

_PROCESS_STARTED_UTC = datetime.now(timezone.utc).isoformat()


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def process_identity():
    """Who wrote a status file; a stale "running" label can be checked against it.

    A PID alone is not proof of liveness (PIDs are reused across reboots).
    """
    return {"pid": os.getpid(), "host": platform.node(), "boot_id": boot_id(),
            "process_started_utc": _PROCESS_STARTED_UTC,
            "written_utc": datetime.now(timezone.utc).isoformat()}


def peak_rss_gib():
    """Peak resident memory of this process, or None where unsupported."""
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # KiB on Linux


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def runtime_versions():
    import importlib.metadata
    import platform
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "networkx", "scikit-learn", "threadpoolctl")}}


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode())


def available_gib():
    if os.name == "nt":
        import ctypes
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available", "virtual", "virtual_available", "extended")]
        data = Memory()
        data.length = ctypes.sizeof(data)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(data)):
            raise OSError("cannot determine available RAM")
        return data.available / 1024**3
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return float(values["MemAvailable"].split()[0]) / 1024**2


def worker_count(requested, *, memory_per_worker_gib, reserve_gib, available=None):
    if not 1 <= requested <= 60:
        raise ValueError("processes must be in 1..60, including the coordinator")
    if (not math.isfinite(memory_per_worker_gib) or memory_per_worker_gib <= 0
            or not math.isfinite(reserve_gib) or reserve_gib < 0):
        raise ValueError("invalid RAM budget")
    free = available_gib() if available is None else available
    slots = int(max(0., free - reserve_gib) // memory_per_worker_gib)
    if slots < 1:
        raise RuntimeError("insufficient available RAM for even one configured rollout worker")
    if requested == 1:
        return 0  # Serial execution in the coordinator; no child process.
    return min(requested - 1, slots, max(1, (os.cpu_count() or 1) - 1))


class RunLock:
    """OS lock releases automatically after a controller crash."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if self.path.stat().st_size == 0:
                self.stream.write(b"0")
                self.stream.flush()
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *args):
        self.stream.close()
