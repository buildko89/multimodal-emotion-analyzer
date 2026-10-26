"""Download and verify model artifacts declared in the model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "models" / "model_manifest.json"
CHUNK_SIZE = 1024 * 1024


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), dict):
        raise ValueError(f"Unsupported model manifest: {path}")
    return manifest


def artifact_path(spec: dict[str, Any]) -> Path:
    destination = (PROJECT_ROOT / spec["destination"]).resolve()
    destination.relative_to(PROJECT_ROOT.resolve())
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(spec: dict[str, Any]) -> tuple[bool, str]:
    path = artifact_path(spec)
    if not path.is_file():
        return False, "missing"
    expected_size = spec.get("size_bytes")
    if expected_size is not None and path.stat().st_size != expected_size:
        return False, f"size mismatch ({path.stat().st_size} != {expected_size})"
    expected_hash = spec.get("sha256")
    if expected_hash:
        actual_hash = sha256_file(path)
        if actual_hash.lower() != expected_hash.lower():
            return False, f"SHA-256 mismatch ({actual_hash})"
    return True, "verified"


def download(name: str, spec: dict[str, Any], force: bool = False) -> None:
    destination = artifact_path(spec)
    valid, detail = verify(spec)
    if valid and not force:
        print(f"[ok] {name}: {destination.relative_to(PROJECT_ROOT)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        print(f"[download] {name}: {spec['url']}")
        request = Request(spec["url"], headers={"User-Agent": "multimodal-emotion-analyzer-model-manager/1"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)

        # Verify the temporary artifact before atomically replacing a working file.
        temporary_spec = dict(spec)
        temporary_spec["destination"] = str(temporary.relative_to(PROJECT_ROOT))
        valid, detail = verify(temporary_spec)
        if not valid:
            raise RuntimeError(f"Downloaded artifact failed verification: {detail}")
        os.replace(temporary, destination)
        print(f"[ok] {name}: {destination.relative_to(PROJECT_ROOT)}")
    finally:
        temporary.unlink(missing_ok=True)


def selected_models(manifest: dict[str, Any], names: list[str], profile: str | None):
    models = manifest["models"]
    if names:
        unknown = sorted(set(names) - set(models))
        if unknown:
            raise KeyError(f"Unknown model(s): {', '.join(unknown)}")
        return [(name, models[name]) for name in names]
    selected_profile = profile or "default"
    return [
        (name, spec)
        for name, spec in models.items()
        if selected_profile in spec.get("required_for", [])
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "check", "download"))
    parser.add_argument("models", nargs="*", help="model IDs; omitted means the selected profile")
    parser.add_argument("--profile", choices=("default", "audio"), help="artifact profile")
    parser.add_argument("--force", action="store_true", help="redownload even when verification succeeds")
    args = parser.parse_args()

    manifest = load_manifest()
    if args.command == "list":
        for name, spec in manifest["models"].items():
            profiles = ",".join(spec.get("required_for", [])) or "optional"
            print(f"{name:30} {spec['role']:32} profiles={profiles}")
        return 0

    selected = selected_models(manifest, args.models, args.profile)
    if not selected:
        parser.error("No models selected")

    failed = False
    for name, spec in selected:
        if args.command == "download":
            download(name, spec, force=args.force)
        valid, detail = verify(spec)
        print(f"[{'ok' if valid else 'error'}] {name}: {detail}")
        failed |= not valid
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
