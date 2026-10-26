"""Compare facial classifications with text rules; report observations only."""

from collections import deque
import math
import time

from models.labels import normalize_emotion_label
from modules.text_sentiment import TextSentimentAnalyzer


class MultiModalAnalyzer:
    """Version 2 observation API. Confidence describes the classifier output."""

    RULE_VERSION = "facial-text-observations-v2"

    def __init__(self, min_confidence=0.3):
        self.text_analyzer = TextSentimentAnalyzer(language="ja")
        self.min_confidence = min_confidence
        self.analysis_history = deque(maxlen=1000)

    def _emotion_only_state(self, emotion_data):
        label = normalize_emotion_label(emotion_data.get("emotion"))
        value = emotion_data.get("confidence")
        valid = isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
        reasons = []
        if label == "unknown":
            reasons.append("unknown_label")
        if not valid:
            reasons.append("invalid_model_confidence")
        elif value < self.min_confidence:
            reasons.append("low_model_confidence")
        quality = emotion_data.get("quality")
        if isinstance(quality, dict) and quality.get("status") == "unavailable":
            reasons.append("input_quality_unavailable")
        return {
            "face_emotion": label, "model_confidence": float(value) if valid else None,
            "input_quality": quality, "input_quality_status": "reported" if quality else "not_measured",
            "status": "unavailable" if reasons else "observed", "uncertainty": reasons,
            "model_id": emotion_data.get("model_id"),
        }

    def analyze(self, emotion_data=None, speech_data=None, timestamp=None):
        face = self._emotion_only_state(emotion_data) if emotion_data else None
        text = speech_data.get("text") if speech_data else None
        text_result = self.text_analyzer.analyze_comprehensive(text) if isinstance(text, str) and text.strip() else None
        reasons = list(face["uncertainty"]) if face else ["no_face_observation"]
        if not text_result:
            reasons.append("no_transcript")
        elif text_result["sentiment"]["polarity"] == "UNKNOWN":
            reasons.append("no_text_polarity_evidence")
        polarity = text_result["sentiment"]["polarity"] if text_result else "UNKNOWN"
        mismatches = []
        if not reasons:
            label = face["face_emotion"]
            expected = {"happy": "POSITIVE", "sad": "NEGATIVE", "angry": "NEGATIVE",
                        "disgust": "NEGATIVE", "fear": "NEGATIVE"}.get(label)
            if expected and polarity in ("POSITIVE", "NEGATIVE") and polarity != expected:
                mismatches.append({
                    "type": "facial_text_polarity_mismatch", "face_emotion": label,
                    "text_polarity": polarity, "description": "表情分類とテキスト極性の不一致候補",
                })
        result = {
            "schema_version": 2, "rule_version": self.RULE_VERSION,
            "timestamp": time.time() if timestamp is None else timestamp,
            "emotion_data": {"emotion": face["face_emotion"], "confidence": face["model_confidence"]} if face else None,
            "speech_data": speech_data, "has_emotion": face is not None, "has_speech": text_result is not None,
            "text_sentiment": text_result,
            "observations": {"face": face, "text_polarity": polarity,
                             "text_method": "japanese_keyword_rules", "acoustic_emotion": "not_implemented"},
            "status": "unavailable" if reasons else "compared", "uncertainty": reasons,
            "signal_mismatches": mismatches,
        }
        self.analysis_history.append(result)
        return result

    def get_history_summary(self, last_n=10):
        recent = list(self.analysis_history)[-last_n:] if last_n > 0 else []
        return {
            "count": len(recent), "compared_count": sum(r["status"] == "compared" for r in recent),
            "unavailable_count": sum(r["status"] == "unavailable" for r in recent),
            "total_signal_mismatches": sum(len(r["signal_mismatches"]) for r in recent),
        }
