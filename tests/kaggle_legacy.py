"""Frozen pre-hardening data from commit 39bb64b, only for historical tests."""

from pathlib import Path
import zipfile
import shutil


ROOT = Path(__file__).resolve().parents[1]


def assets():
    with zipfile.ZipFile(ROOT / "tests/fixtures/kaggle-legacy-data.zip") as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }


def materialize(directory):
    for name, value in assets().items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    return directory / "data"


def build_root(directory):
    materialize(directory)
    for name in ("src", "benchmarks"):
        (directory / name).symlink_to(ROOT / name)
    shutil.copyfile(ROOT / "LICENSE", directory / "LICENSE")
    shutil.copyfile(ROOT / "NOTICE", directory / "NOTICE")
    return directory
