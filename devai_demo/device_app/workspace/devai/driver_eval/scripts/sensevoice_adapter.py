"""Offline research adapter: preserve raw tags; never infer driver state directly."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from prepare_sensevoice import FILES, REVISION, ROOT, WEIGHT_SHA, check_file

TAG_PATTERN = re.compile(r"<\|[^<>]+\|>")
EMOTIONS = {"HAPPY": "happy", "SAD": "sad", "ANGRY": "angry", "NEUTRAL": "neutral",
            "FEARFUL": "fear", "DISGUSTED": "disgust", "SURPRISED": "surprise",
            "OTHER": "other", "EMO_UNKNOWN": "model_unknown"}
LANGUAGES = {"zh", "en", "ja", "ko", "yue", "nospeech"}
EVENTS = {"Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath",
          "Cough", "Sing", "Speech_Noise", "GBG", "Event_UNK"}


def parse_tags(text):
    """Preserve order and unknown tokens; scores are not returned by this backend."""
    tags = TAG_PATTERN.findall(text)
    result = {"raw_text": text, "raw_labels": tags, "raw_scores": None, "confidence": None,
              "asr_text": TAG_PATTERN.sub("", text), "emotion_labels": [],
              "language_labels": [], "audio_event_labels": [], "textnorm_labels": [],
              "unmapped_tags": []}
    for tag in tags:
        name = tag[2:-2]
        if name in EMOTIONS:
            result["emotion_labels"].append(EMOTIONS[name])
        elif name in LANGUAGES:
            result["language_labels"].append(name)
        elif name in EVENTS:
            result["audio_event_labels"].append(name)
        elif name in {"withitn", "woitn"}:
            result["textnorm_labels"].append(name)
        else:
            result["unmapped_tags"].append(tag)
    return result


class SenseVoiceAdapter:
    """Serial, one offline mono 16kHz interval per call; input time is asset time."""
    QUANTIZATIONS = {None, "dynamic_int8_linear"}

    LANGUAGES = {"auto", "ja"}

    def __init__(self, quantization=None, language="auto"):
        if quantization not in self.QUANTIZATIONS:
            raise ValueError("Unsupported SenseVoice quantization")
        if language not in self.LANGUAGES:
            raise ValueError("Unsupported SenseVoice language")
        self.quantization = quantization
        self.language = language
        self.backend = None
        self.provenance = {"model_id": "FunAudioLLM/SenseVoiceSmall", "weight_revision": REVISION,
                           "weight_sha256": WEIGHT_SHA, "weight_verified": False,
                           "runtime_version": importlib.metadata.version("funasr"),
                           "device": "cpu", "trust_remote_code": False, "disable_update": True,
                           "quantization": quantization, "language": language}
        self.load_report = None
        self.diagnostics = {}

    def load(self):
        if self.backend is not None:
            return
        expected = ROOT / "runtime" / "sensevoice" / ".venv"
        if Path(sys.prefix).resolve() != expected.resolve():
            raise RuntimeError("Use the isolated SenseVoice virtual environment")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from funasr import AutoModel
        from funasr.download.download_model_from_hub import download_model
        if torch.__version__ != "2.9.0+cpu" or self.provenance["runtime_version"] != "1.4.16":
            raise RuntimeError("Runtime differs from P1-05 pins")
        start = time.perf_counter()
        model_dir = ROOT / "runtime" / "sensevoice" / "model"
        metadata = json.loads((ROOT / "sensevoice_hf_metadata.json").read_text(encoding="utf-8"))
        if metadata["sha"] != REVISION:
            raise RuntimeError("Metadata revision mismatch")
        entries = {item["rfilename"]: item for item in metadata["siblings"]}
        if entries["model.pt"]["lfs"]["sha256"] != WEIGHT_SHA:
            raise RuntimeError("Weight pin mismatch")
        for name in FILES:
            check_file(model_dir / name, entries[name])
        self.provenance["weight_verified"] = True
        config = download_model(model=str(model_dir), hub="hf", trust_remote_code=False)
        # FunASR's default loader fills missing keys from initialized weights.
        # Load the verified state ourselves with strict=True so none can be missed.
        config.update(init_param=None, device="cpu", ncpu=4, disable_update=True,
                      trust_remote_code=False, disable_pbar=True, disable_log=True)
        backend = AutoModel(**config)
        state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state:
                state = state[key]
        matched = backend.model.load_state_dict(state, strict=True)
        if not all(torch.isfinite(value).all().item() for value in state.values() if torch.is_floating_point(value)):
            raise RuntimeError("Nonfinite model weights")
        self.load_report = {"strict": True, "weights_only": True, "state_keys": len(state),
                            "missing_keys": list(matched.missing_keys), "unexpected_keys": list(matched.unexpected_keys),
                            "weights_finite": True, "verify_and_load_s": time.perf_counter() - start,
                            "threads": torch.get_num_threads()}
        tokenizer = backend.kwargs["tokenizer"]
        self.load_report["vocab_size"] = tokenizer.get_vocab_size()
        self.load_report["special_tokens"] = [
            {"id": i, "token": tokenizer.sp.id_to_piece(i)}
            for i in range(tokenizer.get_vocab_size())
            if TAG_PATTERN.fullmatch(tokenizer.sp.id_to_piece(i))]
        del state
        if self.quantization == "dynamic_int8_linear":
            # Weights stay the verified FP32 file; Linear layers are quantized in memory only.
            quantize_start = time.perf_counter()
            torch.backends.quantized.engine = "qnnpack"
            torch.ao.quantization.quantize_dynamic(backend.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
            self.load_report["quantization"] = {"method": "torch.ao.quantization.quantize_dynamic",
                                                "modules": ["torch.nn.Linear"], "dtype": "qint8",
                                                "engine": torch.backends.quantized.engine,
                                                "seconds": time.perf_counter() - quantize_start}
        def hook(name):
            def record(module, inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                finite = bool(torch.isfinite(tensor).all().item())
                self.diagnostics[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "finite": finite}
                if not finite:
                    raise RuntimeError(f"Nonfinite {name}")
            return record
        backend.model.encoder.register_forward_hook(hook("encoder"))
        backend.model.ctc.ctc_lo.register_forward_hook(hook("ctc_logits"))
        import funasr
        code = Path(funasr.__file__).parent / "models" / "sense_voice" / "model.py"
        self.provenance["runtime_code_sha256"] = hashlib.sha256(code.read_bytes()).hexdigest()
        self.backend = backend

    def analyze(self, samples, *, asset_id, sample_rate=16000, audio_start_s=0.0,
                audio_offset_s=0.0, signal_scope="unverified", speaker="speaker_unverified"):
        received = time.perf_counter()
        result = {"schema": "driver-observation-v0.1", "asset_id": asset_id, "subject_id": None,
                  "interval": None, "mode": "offline", "observations": [], "state_candidates": [],
                  "explicit_preferences": [], "response": None, "status": "not_run",
                  "uncertainty": ["speaker_unverified", "sync_unverified", "state_not_evaluated"],
                  "quality": {"signal_scope": signal_scope, "speaker": speaker,
                              "sync": "not_measured", "emotion_usable_for_driver_state": False},
                  "input": {"sample_rate": sample_rate if isinstance(sample_rate, int) else None,
                            "audio_start_s": audio_start_s if isinstance(audio_start_s, (int, float)) and math.isfinite(audio_start_s) else None,
                            "audio_offset_s": audio_offset_s if isinstance(audio_offset_s, (int, float)) and math.isfinite(audio_offset_s) else None},
                  "model": dict(self.provenance), "backend_output": None, "diagnostics": None,
                  "timing": {"received_monotonic_s": received, "completed_monotonic_s": None}}
        def finish(status, reason=None):
            result["status"] = status
            if reason:
                result["uncertainty"].append(reason)
            done = time.perf_counter()
            result["timing"].update(completed_monotonic_s=done, processing_s=done-received)
            return result
        if samples is None:
            return finish("unavailable", "audio_unavailable")
        if not isinstance(samples, np.ndarray) or samples.ndim != 1:
            return finish("invalid_input", "expected_mono_array")
        if samples.dtype.kind != "f" or not np.isfinite(samples).all():
            return finish("invalid_input", "expected_finite_float_samples")
        if sample_rate != 16000:
            return finish("invalid_input", "expected_16000hz")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (audio_start_s, audio_offset_s)):
            return finish("invalid_input", "invalid_input_time")
        if audio_start_s < 0 or audio_start_s + audio_offset_s < 0:
            return finish("invalid_input", "negative_asset_time")
        duration = len(samples) / sample_rate
        result["input"].update(sample_count=len(samples), duration_s=duration, shape=list(samples.shape))
        if duration == 0:
            return finish("unavailable", "empty_audio")
        result["interval"] = {"start_s": audio_start_s + audio_offset_s,
                              "end_s": audio_start_s + audio_offset_s + duration}
        if not all(math.isfinite(v) for v in result["interval"].values()) or result["interval"]["end_s"] <= result["interval"]["start_s"]:
            result["interval"] = None
            return finish("invalid_input", "invalid_asset_interval")
        if not 0.1 <= duration <= 30.0:
            return finish("invalid_input", "duration_out_of_adapter_bounds")
        if np.abs(samples).max() > 1.0:
            return finish("invalid_input", "samples_outside_normalized_pcm_range")
        result["quality"]["rms"] = float(np.sqrt(np.mean(samples.astype(np.float64)**2)))
        if not np.any(samples):
            result["uncertainty"].append("silent_input")
        if signal_scope != "human_speech_verified":
            result["uncertainty"].append("speech_not_verified")
        try:
            self.load()
            result["model"] = dict(self.provenance)
            self.diagnostics = {}
            start = time.perf_counter()
            raw = self.backend.generate(input=samples.astype(np.float32, copy=False), cache={},
                                        language=self.language, use_itn=False, ban_emo_unk=False,
                                        batch_size=1, output_timestamp=False, disable_pbar=True)
            result["timing"]["generate_s"] = time.perf_counter() - start
            result["backend_output"] = raw
            result["diagnostics"] = dict(self.diagnostics)
            if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0].get("text"), str):
                return finish("backend_error", "unexpected_backend_output")
            if set(self.diagnostics) != {"encoder", "ctc_logits"}:
                return finish("backend_error", "tensor_diagnostics_missing")
            enc, logits = self.diagnostics["encoder"], self.diagnostics["ctc_logits"]
            if (not enc["finite"] or not logits["finite"] or len(enc["shape"]) != 3 or
                    enc["shape"][0] != 1 or enc["shape"][2] != 512 or
                    logits["shape"] != enc["shape"][:2] + [25055]):
                return finish("backend_error", "invalid_tensor_diagnostics")
            parsed = parse_tags(raw[0]["text"])
            parsed.update(kind="sensevoice_audio", interval=dict(result["interval"]))
            result["observations"].append(parsed)
            if parsed["unmapped_tags"]:
                result["uncertainty"].append("unmapped_model_tags")
            if not parsed["emotion_labels"]:
                result["uncertainty"].append("emotion_tag_missing")
            return finish("ok")
        except Exception as exc:
            result["model"] = dict(self.provenance)
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            return finish("backend_error", "model_execution_failed")
