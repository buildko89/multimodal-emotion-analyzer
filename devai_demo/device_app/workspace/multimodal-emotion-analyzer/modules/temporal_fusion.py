"""Join timestamped observations within half-open speech intervals."""
import math
from collections import Counter
from models.labels import normalize_emotion_label


def speech_windows(result, origin_ns, offset_seconds=0.0):
    windows = []
    for index, segment in enumerate(result.get("segments", [])):
        start, end = float(segment["start"]), float(segment["end"])
        if not all(math.isfinite(x) for x in (start, end, offset_seconds)) or start < 0 or end <= start:
            raise ValueError("Speech segments require finite 0 <= start < end")
        windows.append({
            "segment_id": index, "start_ns": origin_ns + round((start + offset_seconds) * 1e9),
            "end_ns": origin_ns + round((end + offset_seconds) * 1e9),
            "text": segment.get("text", ""), "speaker_id": None,
            "language": result.get("language"),
        })
    return windows


class WindowJoiner:
    """Accumulate only one-person observations; never choose the first face."""
    def __init__(self, windows, min_confidence=0.3):
        self.windows = windows
        self.min_confidence = min_confidence
        self.samples = {w["segment_id"]: [] for w in windows}
        self.blocked = {w["segment_id"]: set() for w in windows}

    def observe(self, timestamp_ns, faces):
        for window in self.windows:
            if window["start_ns"] <= timestamp_ns < window["end_ns"]:
                key = window["segment_id"]
                if len(faces) != 1:
                    self.blocked[key].add("multiple_faces" if faces else "no_face")
                else:
                    face = dict(faces[0])
                    face["emotion"] = normalize_emotion_label(face.get("emotion"))
                    confidence = face.get("confidence")
                    if face["emotion"] == "unknown":
                        self.blocked[key].add("unknown_label")
                    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        self.blocked[key].add("invalid_model_confidence")
                        face["confidence"] = 0.0
                    elif confidence < self.min_confidence:
                        self.blocked[key].add("low_model_confidence")
                    quality = face.get("quality")
                    if isinstance(quality, dict) and quality.get("status") == "unavailable":
                        self.blocked[key].add("input_quality_unavailable")
                    self.samples[key].append(face)

    def results(self, processed_until_ns):
        for window in self.windows:
            key = window["segment_id"]
            samples = self.samples[key]
            reasons = set(self.blocked[key])
            ids = {s["face_id"] for s in samples}
            if len(ids) > 1:
                reasons.add("track_changed")
            if not samples:
                reasons.add("no_observations")
            if window["end_ns"] > processed_until_ns:
                reasons.add("incomplete_window")
            counts = Counter(s["emotion"] for s in samples)
            label = counts.most_common(1)[0][0] if counts else "unknown"
            if len(counts) > 1 and counts.most_common(2)[0][1] == counts.most_common(2)[1][1]:
                reasons.add("ambiguous_face_label")
            confidence = (sum(s["confidence"] for s in samples if s["emotion"] == label)
                          / counts[label]) if counts else 0.0
            if label == "unknown":
                reasons.add("unknown_label")
            yield {
                "schema_version": 2,
                "window_start_ns": window["start_ns"], "window_end_ns": window["end_ns"],
                "speech_data": window, "track_id": next(iter(ids)) if len(ids) == 1 else None,
                "status": "unavailable" if reasons else "aligned",
                "uncertainty": sorted(reasons), "sample_count": len(samples),
                "emotion_distribution": {k: v / len(samples) for k, v in counts.items()},
                "emotion_data": {
                    "emotion": label, "confidence": confidence,
                    "model_id": samples[0].get("model_id"),
                    "quality": None,
                } if not reasons else None,
                "speaker_association": "single_visible_person_assumption",
            }
