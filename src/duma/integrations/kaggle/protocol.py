"""Versioned, credential-free configuration for Kaggle episodes."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str | None = Field(default=None, min_length=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=2048, gt=0)
    reasoning: Literal["none", "low", "medium", "high"] | None = None

    def generation_args(self) -> dict:
        return self.model_dump(exclude={"model"}, exclude_none=True)


class KaggleProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    user: ModelSettings
    judge: ModelSettings
    agent: ModelSettings = Field(default_factory=ModelSettings)
    regimes: list[Literal["solo", "dual"]]
    trial_seeds: list[int]
    max_steps: int = Field(default=100, gt=0)
    max_errors: int = Field(default=10, gt=0)
    tasks: dict[str, list[str]]
    task_data_sha256: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_manifest(self):
        for values, label in [
            (self.regimes, "regimes"),
            (self.trial_seeds, "trial seeds"),
        ]:
            if not values or len(values) != len(set(values)):
                raise ValueError(f"Expected nonempty, unique {label}")
        if not self.tasks:
            raise ValueError("Task manifest must not be empty")
        for domain, ids in self.tasks.items():
            if not ids or len(ids) != len(set(ids)):
                raise ValueError(f"Expected nonempty, unique task IDs for {domain}")
        if self.agent.model is not None:
            raise ValueError("The target model must be selected by Kaggle")
        return self

    def require_service_models(self) -> None:
        if not self.user.model or not self.judge.model:
            raise ValueError("Pin user.model and judge.model before running inference")
        if self.user.model == self.judge.model:
            raise ValueError("Select separate user and judge models for this protocol")

    def validate_data(self, data_dir: Path) -> None:
        from duma.integrations.kaggle.task_revision import validate_revision

        validate_revision(self.version, data_dir)
        if self.version in {
            "duma-kaggle-private-full-v3",
            "duma-kaggle-private-pilot-v6",
        }:
            if not self.task_data_sha256:
                raise ValueError("Hardened protocol requires pinned task data hashes")
            actual = {
                path.relative_to(data_dir).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in (data_dir / "duma").rglob("*")
                if path.is_file()
                and path.suffix in {".json", ".md"}
                and path.name != "kaggle-task-revision.json"
            }
            if actual != self.task_data_sha256:
                raise ValueError("Hardened task data hash mismatch")
        for domain, task_ids in self.tasks.items():
            path = data_dir / "duma" / "domains" / domain / "tasks.json"
            available = {task["id"] for task in json.loads(path.read_text())}
            missing = set(task_ids) - available
            if missing:
                raise ValueError(f"Unknown tasks in {domain}: {sorted(missing)}")

    @property
    def fingerprint(self) -> str:
        data = self.model_dump()
        if self.task_data_sha256 is None:
            data.pop("task_data_sha256")  # Preserve historical protocol identities.
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_protocol(path: str | Path) -> KaggleProtocol:
    return KaggleProtocol.model_validate_json(Path(path).read_text())


def fingerprint_files(root: Path, patterns: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    paths = sorted({path for pattern in patterns for path in root.glob(pattern)})
    if not paths:
        raise ValueError(f"No protocol assets found under {root}")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()
