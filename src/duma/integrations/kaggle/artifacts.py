"""Atomic, checksummed episode checkpoints; no provider dependencies."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def checksum(artifact: dict) -> str:
    body = {key: value for key, value in artifact.items() if key != "artifact_hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def read_checkpoint(path: Path, metadata: dict) -> dict:
    artifact = json.loads(path.read_text())
    if not isinstance(artifact, dict):
        raise ValueError("Checkpoint must be a JSON object")
    if artifact.get("metadata") != metadata or artifact.get(
        "artifact_hash"
    ) != checksum(artifact):
        raise ValueError(f"Checkpoint identity/integrity mismatch: {path.name}")
    if artifact.get("status") == "completed":
        score = artifact.get("score")
        if (
            score not in (0.0, 1.0)
            or artifact.get("simulation", {}).get("reward_info", {}).get("reward")
            != score
        ):
            raise ValueError("Invalid completed checkpoint score")
    return artifact


def write_checkpoint(path: Path, artifact: dict):
    artifact["artifact_hash"] = checksum(artifact)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(artifact, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def episode_lock(path: Path):
    lock = path.with_suffix(".lock")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)
