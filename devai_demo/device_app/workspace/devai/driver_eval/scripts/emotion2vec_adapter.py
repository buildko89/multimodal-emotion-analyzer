"""Pinned research SER backend: raw 9-way scores, no driver state inference."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from prepare_additional_models import EMOTION_REV, EMOTION_SHA
from prepare_sensevoice import ROOT, check_file, sha256

LABELS = ["生气/angry", "厌恶/disgusted", "恐惧/fearful", "开心/happy", "中立/neutral",
          "其他/other", "难过/sad", "吃惊/surprised", "<unk>"]
NORMALIZED = ["angry", "disgust", "fear", "happy", "neutral", "other", "sad", "surprise", "model_unknown"]


class Emotion2vecAdapter:
    def __init__(self):
        self.backend = None
        self.load_report = None
        self.diagnostics = {}
        self.provenance = {"model_id": "emotion2vec/emotion2vec_plus_base", "weight_revision": EMOTION_REV,
                           "weight_sha256": EMOTION_SHA, "runtime_version": importlib.metadata.version("funasr"),
                           "device": "cpu", "weight_verified": False}

    def load(self):
        if self.backend is not None:
            return
        if Path(__import__("sys").prefix).resolve() != (ROOT / "runtime" / "sensevoice" / ".venv").resolve():
            raise RuntimeError("Use the shared isolated audio research environment")
        for package, version in (("funasr", "1.4.16"), ("torch", "2.9.0"), ("torchaudio", "2.9.0")):
            if importlib.metadata.version(package) != version:
                raise RuntimeError(f"Unexpected runtime version: {package}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        if torch.__version__ != "2.9.0+cpu" or torch.version.cuda is not None:
            raise RuntimeError("Expected ARM CPU torch runtime")
        from funasr import AutoModel
        from funasr.download.download_model_from_hub import download_model
        start = time.perf_counter()
        folder = ROOT / "runtime" / "emotion2vec" / "model"
        metadata = json.loads((ROOT / "additional_model_evidence" / "emotion_hf_metadata.json").read_text(encoding="utf-8"))
        if metadata["sha"] != EMOTION_REV or sha256(folder / "model.pt") != EMOTION_SHA:
            raise RuntimeError("Emotion weight integrity mismatch")
        entries = {e["rfilename"]: e for e in metadata["siblings"]}
        for name in ("model.pt", "config.yaml", "configuration.json", "tokens.txt", "README.md"):
            check_file(folder / name, entries[name])
        config = download_model(model=str(folder), hub="hf", trust_remote_code=False)
        config["model_conf"]["modalities"]["audio"]["decoder"] = None
        config.update(init_param=None, device="cpu", ncpu=4, disable_update=True, trust_remote_code=False,
                      disable_pbar=True, disable_log=True)
        backend = AutoModel(**config)
        state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state:
                state = state[key]
        # The official checkpoint wraps backbone keys with d2v_model.; the
        # installed FunASR implementation exposes that backbone at its root.
        state = {key.removeprefix("d2v_model."): value for key, value in state.items()}
        if not all(torch.isfinite(tensor).all().item() for tensor in state.values() if torch.is_tensor(tensor)):
            raise RuntimeError("Nonfinite inference weights")
        matched = backend.model.load_state_dict(state, strict=True)
        self.load_report = {"strict": True, "state_keys": len(state), "missing_keys": list(matched.missing_keys),
                            "unexpected_keys": list(matched.unexpected_keys), "load_s": time.perf_counter()-start,
                            "key_mapping": "remove d2v_model. prefix; strict match all inference keys",
                            "training_only_decoder_disabled": True}
        self.load_report["weights_finite"] = True
        del state
        if backend.kwargs["tokenizer"].token_list != LABELS:
            raise RuntimeError("Classifier label order mismatch")
        def hook(module, inputs, output):
            finite = bool(torch.isfinite(output).all().item())
            self.diagnostics["classifier_logits"] = {"shape": list(output.shape), "finite": finite}
            if not finite:
                raise RuntimeError("Nonfinite classifier logits")
        backend.model.proj.register_forward_hook(hook)
        self.provenance.update(weight_verified=True, runtime_code_sha256=sha256(Path(__import__("funasr").__file__).parent / "models" / "emotion2vec" / "model.py"))
        self.backend = backend

    def analyze(self, samples, *, asset_id, sample_rate=16000, audio_start_s=0.0, audio_offset_s=0.0,
                signal_scope="unverified", speaker="speaker_unverified"):
        received = time.perf_counter()
        result = {"schema": "driver-observation-v0.1", "asset_id": asset_id, "interval": None,
                  "mode": "offline", "observations": [], "state_candidates": [], "response": None,
                  "status": "not_run", "model": dict(self.provenance), "backend_output": None,
                  "uncertainty": ["speaker_unverified", "sync_unverified", "state_not_evaluated"],
                  "quality": {"signal_scope": signal_scope, "speaker": speaker, "sync": "not_measured",
                              "emotion_usable_for_driver_state": False},
                  "timing": {"received_monotonic_s": received}}
        def finish(status, reason=None):
            result["status"] = status
            if reason:
                result["uncertainty"].append(reason)
            done = time.perf_counter()
            result["timing"].update(completed_monotonic_s=done, processing_s=done-received)
            return result
        if samples is None:
            return finish("unavailable", "audio_unavailable")
        if not isinstance(samples, np.ndarray) or samples.ndim != 1 or samples.dtype.kind != "f" or not np.isfinite(samples).all():
            return finish("invalid_input", "expected_finite_mono_float")
        if sample_rate != 16000 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (audio_start_s, audio_offset_s)):
            return finish("invalid_input", "invalid_rate_or_time")
        if len(samples) == 0:
            return finish("unavailable", "empty_audio")
        duration = len(samples)/sample_rate
        left, right = audio_start_s+audio_offset_s, audio_start_s+audio_offset_s+duration
        if (audio_start_s < 0 or not 0 <= left < right or not math.isfinite(right) or
                not 0.1 <= duration <= 30 or np.abs(samples).max() > 1):
            return finish("invalid_input", "invalid_range_or_interval")
        result["interval"] = {"start_s": left, "end_s": right}
        result["input"] = {"sample_count": len(samples), "sample_rate": sample_rate, "duration_s": duration}
        if not np.any(samples):
            result["uncertainty"].append("silent_input")
        try:
            self.load()
            self.diagnostics = {}
            result["model"] = dict(self.provenance)
            start = time.perf_counter()
            raw = self.backend.generate(input=samples.astype(np.float32, copy=False), cache={},
                                        granularity="utterance", extract_embedding=True, batch_size=1, disable_pbar=True)
            result["timing"]["generate_s"] = time.perf_counter()-start
            if not isinstance(raw, list) or len(raw) != 1 or raw[0]["labels"] != LABELS:
                raise RuntimeError("Unexpected classifier output")
            scores = np.asarray(raw[0]["scores"])
            embedding = np.asarray(raw[0]["feats"])
            if (scores.shape != (9,) or embedding.shape != (768,) or not np.isfinite(scores).all() or
                    not np.isfinite(embedding).all() or np.any(scores < 0) or np.any(scores > 1) or
                    abs(float(scores.sum())-1) > 1e-5):
                raise RuntimeError("Invalid score/embedding tensor")
            self.diagnostics["embedding"] = {"shape": list(embedding.shape), "finite": True}
            result["diagnostics"] = self.diagnostics
            result["backend_output"] = [{"key": raw[0]["key"], "labels": raw[0]["labels"],
                                          "scores": scores.tolist(), "feats": embedding.tolist()}]
            result["observations"] = [{"kind": "emotion2vec_audio", "raw_labels": LABELS,
                                       "raw_scores": scores.tolist(), "normalized_labels": NORMALIZED,
                                       "top_label": NORMALIZED[int(scores.argmax())], "confidence": None,
                                       "scores_calibrated": False, "embedding": embedding.tolist(),
                                       "interval": dict(result["interval"])}]
            return finish("ok")
        except Exception as error:
            result["error"] = {"type": type(error).__name__, "message": str(error)}
            return finish("backend_error", "model_execution_failed")
