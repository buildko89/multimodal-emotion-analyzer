"""Small dependency-free tracker used to keep face IDs stable between frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

BBox = tuple[int, int, int, int]


def intersection_over_union(left: BBox, right: BBox) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


@dataclass
class _Track:
    bbox: BBox
    missed_frames: int = 0


class IoUTracker:
    """Greedy IoU association suitable for the initial single/few-person scope."""

    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 5):
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0

    def update(self, detections: Iterable[BBox]) -> list[int]:
        boxes = list(detections)
        assignments: list[int | None] = [None] * len(boxes)
        candidates = sorted(
            (
                (intersection_over_union(track.bbox, box), track_id, detection_index)
                for track_id, track in self._tracks.items()
                for detection_index, box in enumerate(boxes)
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for score, track_id, detection_index in candidates:
            if score < self.iou_threshold:
                break
            if track_id in used_tracks or detection_index in used_detections:
                continue
            assignments[detection_index] = track_id
            used_tracks.add(track_id)
            used_detections.add(detection_index)
            self._tracks[track_id] = _Track(boxes[detection_index])

        for track_id in list(self._tracks):
            if track_id in used_tracks:
                continue
            self._tracks[track_id].missed_frames += 1
            if self._tracks[track_id].missed_frames > self.max_missed_frames:
                del self._tracks[track_id]

        for detection_index, box in enumerate(boxes):
            if assignments[detection_index] is not None:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track(box)
            assignments[detection_index] = track_id

        return [int(track_id) for track_id in assignments]

