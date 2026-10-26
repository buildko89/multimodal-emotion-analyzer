"""Factories for configuration-selected inference backends."""

from __future__ import annotations

from models.detector import EmotionDetector


def create_emotion_detector(config: dict) -> EmotionDetector:
    """Create the detector selected by config without silently overriding it."""
    face_backend = config.get("face_detection", {}).get("model")
    if face_backend == "mediapipe":
        # The subclass uses the current MediaPipe Tasks API and a local model.
        from emotion_detector_mediapipe import MediaPipeEmotionDetector

        return MediaPipeEmotionDetector(config)
    return EmotionDetector(config)

