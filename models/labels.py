"""Canonical output labels shared by all emotion backends."""

CANONICAL_EMOTIONS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "surprised": "surprise",
    "surprise": "surprise",
    "neutral": "neutral",
}


def normalize_emotion_label(label: str) -> str:
    """Normalize known backend spellings; do not guess unknown class mappings."""
    return _ALIASES.get(str(label).strip().lower(), "unknown")


def canonical_score_map(labels, scores):
    """Map declared backend classes without guessing their order or dropping mass."""
    if len(labels) != len(scores):
        raise ValueError("Backend class labels and scores must have equal lengths")
    result = {label: 0.0 for label in CANONICAL_EMOTIONS}
    result["unknown"] = 0.0
    for label, score in zip(labels, scores):
        result[normalize_emotion_label(label)] += float(score)
    return result
