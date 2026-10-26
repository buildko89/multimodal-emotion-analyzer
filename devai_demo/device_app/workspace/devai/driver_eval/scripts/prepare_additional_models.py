"""Fetch pinned data files only and retain notices for local research evaluation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

from prepare_sensevoice import ROOT, check_file, sha256

EMOTION_REV = "b318240bfe67db81a8c572ecb37ce9c3759b81c9"
EMOTION_SHA = "60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd"
FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_SHA = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def download(url, path):
    if path.exists():
        return
    partial = path.with_suffix(path.suffix+".partial")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
        count = 0
        last = 0
        for chunk in iter(lambda: response.read(1024*1024), b""):
            stream.write(chunk)
            count += len(chunk)
            if count-last >= 256*1024*1024:
                print(f"{path.name}: {count} bytes", flush=True)
                last = count
    partial.replace(path)


def main():
    evidence = ROOT / "additional_model_evidence"
    face = ROOT / "runtime" / "face_landmarker" / "model"
    emotion = ROOT / "runtime" / "emotion2vec" / "model"
    face.mkdir(parents=True, exist_ok=True)
    emotion.mkdir(parents=True, exist_ok=True)
    records = []
    path = face / "face_landmarker.task"
    print("Acquiring official version 1 Face Landmarker bundle", flush=True)
    download(FACE_URL, path)
    if sha256(path) != FACE_SHA:
        raise ValueError("Face version 1 bytes differ from reviewed artifact")
    with zipfile.ZipFile(path) as bundle:
        submodels = [{"name": name, "size": len(bundle.read(name)), "sha256": hashlib.sha256(bundle.read(name)).hexdigest()}
                     for name in bundle.namelist()]
    records.append({"model": "MediaPipe Face Landmarker", "version": "1", "path": str(path.relative_to(ROOT)),
                    "source": FACE_URL, "sha256": sha256(path), "size": path.stat().st_size, "submodels": submodels})
    for name in ("face_detector_card.pdf", "face_mesh_card.pdf", "blendshape_card.pdf", "apache_license.txt"):
        shutil.copyfile(evidence / name, face / name)
    metadata = json.loads((evidence / "emotion_hf_metadata.json").read_text(encoding="utf-8"))
    if metadata["sha"] != EMOTION_REV:
        raise ValueError("Unexpected emotion revision")
    entries = {e["rfilename"]: e for e in metadata["siblings"]}
    if entries["model.pt"]["lfs"]["sha256"] != EMOTION_SHA:
        raise ValueError("Unexpected emotion weight hash")
    for name in ("model.pt", "config.yaml", "configuration.json", "tokens.txt", "README.md"):
        url = f"https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/{EMOTION_REV}/{name}"
        path = emotion / name
        print(f"Acquiring emotion2vec+: {name}", flush=True)
        download(url, path)
        digest = check_file(path, entries[name])
        records.append({"model": "emotion2vec_plus_base", "revision": EMOTION_REV,
                        "source": url, "path": str(path.relative_to(ROOT)), "size": path.stat().st_size,
                        "sha256": digest})
    shutil.copyfile(ROOT / "sensevoice_license_evidence_20261004_final" / "referenced_MODEL_LICENSE.txt", emotion / "MODEL_LICENSE")
    shutil.copyfile(evidence / "apache_license.txt", emotion / "APACHE_LICENSE")
    for folder, text in ((face, "MediaPipe Face Landmarker version 1; Google / MediaPipe Authors.\nOfficial model cards and Apache License 2.0 retained. Research observations only; no calibrated drowsiness decision.\n"),
                         (emotion, "emotion2vec_plus_base; Ziyang Ma, Zhisheng Zheng, Jiaxin Ye, Jinchao Li, Zhifu Gao, Shiliang Zhang, Xie Chen; emotion2vec / Alibaba Tongyi Lab.\nHF model-license and ModelScope Apache 2.0 labels differ. Retain both terms and sources; local research only pending deployment clarification.\n")):
        (folder / "ATTRIBUTION.txt").write_text(text, encoding="utf-8")
    result = {"acquired_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "local PC research",
              "files": records, "remote_code_downloaded": False}
    (ROOT / "additional_model_manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print("Official model data acquired and verified.", flush=True)


if __name__ == "__main__":
    main()
