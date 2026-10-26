"""
モジュールパッケージ初期化
"""

from .text_sentiment import TextSentimentAnalyzer

__all__ = ["WhisperRecognizer", "TextSentimentAnalyzer"]


def __getattr__(name):
    """Keep the optional Whisper dependency lazy for video-only use."""
    if name == "WhisperRecognizer":
        from .whisper_recognizer import WhisperRecognizer

        return WhisperRecognizer
    raise AttributeError(name)
