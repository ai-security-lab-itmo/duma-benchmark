"""Build a credential-free private Dataset and its hash-pinned hosted task."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def build_bundle(root: Path, output: Path, protocol: Path, owner: str) -> str:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory for a new bundle")
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        root / name for name in ("pyproject.toml", "README.md", "LICENSE", "uv.lock")
    ]
    paths += sorted((root / "src" / "duma").rglob("*.py"))
    paths += sorted((root / "data" / "duma").rglob("*.json"))
    paths += sorted((root / "data" / "duma").rglob("*.md"))
    contents = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected an ordinary source file: {path}")
        contents[path.relative_to(root).as_posix()] = path.read_bytes()
    contents["protocol.json"] = protocol.read_bytes()
    file_hashes = {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }
    contents["bundle-files.json"] = json.dumps(
        file_hashes, indent=2, sort_keys=True
    ).encode()
    bundle = output / "duma-pilot.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(contents["bundle-files.json"]).hexdigest()
    template = (root / "benchmarks/kaggle/hosted_pilot.py").read_text()
    (output / "hosted_pilot.py").write_text(
        template.replace("__DUMA_BUNDLE_SHA256__", digest).replace(
            "__DUMA_MANIFEST_SHA256__", manifest_digest
        )
    )
    metadata = {
        "title": "DUMA Bench Private Pilot Runtime",
        "id": f"{owner}/duma-bench-private-pilot-runtime",
        "licenses": [{"name": "MIT"}],
        "description": "Private DUMA pilot code, original task assets and pinned pilot protocol. No model credentials or experiment results are included.",
    }
    (output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "bundle-sha256.txt").write_text(digest + "\n")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("protocol.pilot.v1.json"),
    )
    args = parser.parse_args()
    print(
        build_bundle(
            Path(__file__).resolve().parents[2], args.output, args.protocol, args.owner
        )
    )
