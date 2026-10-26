"""Acquire only pinned SenseVoiceSmall data; never execute repository Python."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
REPO = "FunAudioLLM/SenseVoiceSmall"
FILES = ("model.pt", "config.yaml", "configuration.json", "am.mvn",
         "chn_jpn_yue_eng_ko_spectok.bpe.model", "README.md")
WEIGHT_SHA = "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "driver-eval-P1-05"}), timeout=120)


def check_file(path, entry):
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"Size mismatch: {path.name}")
    digest = sha256(path)
    expected = entry.get("lfs", {}).get("sha256")
    if expected and digest != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    # Small non-LFS files are identified by immutable Git blob ID.
    if not expected:
        data = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != entry["blobId"]:
            raise ValueError(f"Git blob mismatch: {path.name}")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    runtime = ROOT / "runtime" / "sensevoice"
    model = runtime / "model"
    metadata_path = ROOT / "sensevoice_hf_metadata.json"
    if args.verify_only:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
        with fetch(url) as response:
            metadata = json.load(response)
        if metadata["sha"] != REVISION:
            raise ValueError("Unexpected model revision")
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if metadata["sha"] != REVISION:
        raise ValueError("Unexpected cached revision")
    entries = {item["rfilename"]: item for item in metadata["siblings"]}
    if entries["model.pt"]["lfs"]["sha256"] != WEIGHT_SHA:
        raise ValueError("Weight hash differs from reviewed candidate")
    model.mkdir(parents=True, exist_ok=True)
    records = []
    for filename in FILES:
        entry = entries[filename]
        path = model / filename
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}"
        if not path.exists() and not args.verify_only:
            partial = path.with_suffix(path.suffix + ".partial")
            print(f"Downloading {filename}: {entry['size']} bytes", flush=True)
            with fetch(url) as response, partial.open("wb") as stream:
                downloaded = 0
                last_report = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_report >= 128 * 1024 * 1024:
                        print(f"  {filename}: {downloaded} bytes", flush=True)
                        last_report = downloaded
            check_file(partial, entry)
            partial.replace(path)
        digest = check_file(path, entry)
        records.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "source": url, "size": path.stat().st_size,
                        "sha256": digest, "expected_lfs_sha256": entry.get("lfs", {}).get("sha256"),
                        "git_blob_id": entry["blobId"]})
        print(f"Verified {filename}: {digest}", flush=True)
    evidence = ROOT / "sensevoice_license_evidence_20261004_final"
    for source, name in (("referenced_MODEL_LICENSE.txt", "MODEL_LICENSE"),
                         ("code_LICENSE.txt", "SenseVoice_CODE_LICENSE"),
                         ("official_readme.txt", "SenseVoice_README.txt")):
        if not args.verify_only:
            shutil.copyfile(evidence / source, model / name)
        if sha256(evidence / source) != sha256(model / name):
            raise ValueError(f"Notice mismatch: {name}")
        records.append({"path": str((model / name).relative_to(ROOT)).replace("\\", "/"),
                        "evidence_source": str((evidence / source).relative_to(ROOT)).replace("\\", "/"),
                        "size": (model / name).stat().st_size, "sha256": sha256(model / name)})
    attribution = model / "ATTRIBUTION.md"
    if not args.verify_only:
        shutil.copyfile(ROOT / "SENSEVOICE_ATTRIBUTION.md", attribution)
    if sha256(attribution) != sha256(ROOT / "SENSEVOICE_ATTRIBUTION.md"):
        raise ValueError("Attribution mismatch")
    records.append({"path": str(attribution.relative_to(ROOT)).replace("\\", "/"),
                    "size": attribution.stat().st_size, "sha256": sha256(attribution)})
    report = {"task": "P1-05", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "model": REPO, "revision": REVISION, "files": records,
              "remote_python_downloaded": False, "inference_run": False,
              "scope": "local PC research evaluation; see SENSEVOICE_LICENSE_REVIEW.md"}
    (ROOT / "sensevoice_model_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Model and notices verified. No model inference performed.", flush=True)


if __name__ == "__main__":
    main()
