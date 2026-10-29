"""Fixed scientific scope, split guards, atomic artifacts and CPU admission."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

DEVELOPMENT = frozenset(("tel_aviv", "haifa", "manhattan", "new_delhi", "delft", "leuven", "oxford", "palo_alto"))
HELD_OUT = frozenset(("cambridge_uk", "ghent", "salamanca", "lucca"))
POLICIES = ("myopic_ab", "reactive_insertion", "paper_sa_adapted", "full_no_nn", "full_no_idle",
            "full", "full_queue_aware", "full_coordinated_idle", "queue_mle_reservation", "full_mle_reservation")
NO_FULL_MLE_POLICIES = tuple(p for p in POLICIES if p != "full_mle_reservation")


def validate_policies(policies):
    """Preserve the original experiment and the explicit nine-action ablation."""
    policies = tuple(policies)
    if policies not in (POLICIES, NO_FULL_MLE_POLICIES):
        raise ValueError("Unsupported candidate portfolio or action ordering")
    return policies


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value):
    durable_bytes(path, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())


def sync_directory(path):
    """Persist a rename on POSIX. Windows fsyncs the file before replacement."""
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def durable_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        sync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError("Refusing to replace immutable artifact: " + str(path))
    else:
        atomic(path, value)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def guard_city(city, split):
    if split in ("train", "validation") and city not in DEVELOPMENT:
        raise ValueError(f"{city!r} is forbidden in {split}; test cities stay held out")
    if split == "test" and city not in HELD_OUT:
        raise ValueError("Final testing accepts only the four reserved cities")
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown scenario split")


@dataclass(frozen=True)
class Config:
    backend_root: str
    map_manifest: str
    output: str
    spatial_model: str
    paper_model: str
    dependency_dir: str = ""
    decision_minutes: int = 15
    duration_hours: int = 12
    train_per_city: int = 100
    validation_per_city: int = 25
    test_per_city: int = 50
    base_seed: int = 2026102501
    worker_limit: int = 64
    reserved_cpus: int = 8
    fit_threads: int = 4
    memory_reserve_gib: float = 64.
    worker_memory_gib: float = 1.5
    trainer_memory_gib: float = 8.
    cancellation_grace_seconds: float = 15.
    fit_timeout_seconds: float = 3600.  # Historical config field; no elapsed-time fit abort.
    update_every_states: int = 2000
    bootstrap_policy: str = "reactive_insertion"
    max_post_horizon_minutes: int = 10080
    policies: tuple[str, ...] = POLICIES

    def __post_init__(self):
        if self.decision_minutes != 15 or self.duration_hours != 12:
            raise ValueError("This experiment fixes 15-minute decisions and 12-hour scenarios")
        for name in ("train_per_city", "validation_per_city", "test_per_city", "worker_limit", "fit_threads", "update_every_states", "max_post_horizon_minutes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.reserved_cpus, bool) or not isinstance(self.reserved_cpus, int) or self.reserved_cpus < self.fit_threads + 1:
            raise ValueError("Reserve CPU capacity for the trainer and coordinator")
        if (not math.isfinite(self.worker_memory_gib) or self.worker_memory_gib <= 0
                or not math.isfinite(self.memory_reserve_gib) or self.memory_reserve_gib < 64
                or not math.isfinite(self.trainer_memory_gib) or self.trainer_memory_gib <= 0):
            raise ValueError("Invalid memory budget; retain at least 64 GiB reserve")
        if any(not math.isfinite(v) or v <= 0 for v in (self.cancellation_grace_seconds, self.fit_timeout_seconds)):
            raise ValueError("Shutdown and fitting time limits must be positive")
        validate_policies(self.policies)
        if self.bootstrap_policy not in self.policies:
            raise ValueError("Bootstrap policy must belong to the candidate portfolio")

    @classmethod
    def load(cls, path):
        data = read(path)
        if "policies" in data:
            data["policies"] = tuple(data["policies"])
        return cls(**data)

    def maps(self, split):
        manifest = read(self.map_manifest)
        if manifest.get("status") != "complete":
            raise ValueError("Map preparation is incomplete")
        rows = manifest["maps"]
        if len({r['key'] for r in rows}) != len(rows):
            raise ValueError("Duplicate map identities")
        if {r["key"] for r in rows if r["role"] == "development"} != DEVELOPMENT:
            raise ValueError("Development-map roles disagree with the frozen eight-city allowlist")
        if {r["key"] for r in rows if r["role"] == "reserved_final_test"} != HELD_OUT:
            raise ValueError("Reserved-city roles disagree with the four-city holdout")
        result = [r for r in rows if r["key"] in (HELD_OUT if split == "test" else DEVELOPMENT)]
        for row in result:
            guard_city(row["key"], split)
        return result


def available_gib():
    path = Path("/proc/meminfo")
    if not path.exists():
        raise RuntimeError("Collection admission requires the Linux VM's available-memory reading")
    values = dict(line.split(":", 1) for line in path.read_text().splitlines())
    return float(values["MemAvailable"].split()[0]) / 1024**2


def workers(config, cpus=None, memory_gib=None):
    cpus = cpus if cpus is not None else (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
    memory_gib = available_gib() if memory_gib is None else memory_gib
    cpu_slots = max(0, cpus - config.reserved_cpus)
    memory_slots = max(0, int((memory_gib - config.memory_reserve_gib - config.trainer_memory_gib) // config.worker_memory_gib))
    return min(config.worker_limit, cpu_slots, memory_slots)
