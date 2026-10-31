"""Decode MediaPipe short-range BlazeFace outputs (128x128, 896 anchors).

Anchor layout and box decoding follow MediaPipe's face_detection_short_range
graph: strides 8/16/16/16, two anchors per cell at stride 8 and six at
stride 16, fixed unit anchor size, coordinates scaled by 128, sigmoid scores,
then non-maximum suppression at IoU 0.3.
"""
import numpy as np

INPUT_SIZE = 128


def anchors():
    rows = []
    for stride, per_cell in ((8, 2), (16, 6)):
        cells = INPUT_SIZE // stride
        for y in range(cells):
            for x in range(cells):
                rows.extend([((x + 0.5) / cells, (y + 0.5) / cells)] * per_cell)
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (896, 2):
        raise AssertionError('blazeface_anchor_count')
    return result


ANCHORS = anchors()


def letterbox(rgb):
    """Pad to a square (as MediaPipe does) and resize to 128x128 uint8 RGB."""
    import cv2
    height, width = rgb.shape[:2]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    square = np.zeros((side, side, 3), dtype=np.uint8)
    square[top:top + height, left:left + width] = rgb
    return cv2.resize(square, (INPUT_SIZE, INPUT_SIZE)), (side, top, left)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decode(regressors, logits, geometry, image_shape, min_score=0.5, nms_iou=0.3):
    """Return [(x1, y1, x2, y2, score)] in image pixels and the best raw score."""
    reg = np.asarray(regressors, dtype=np.float32).reshape(896, 16)
    raw = np.clip(np.asarray(logits, dtype=np.float32).reshape(896), -100, 100)
    scores = 1.0 / (1.0 + np.exp(-raw))
    best = float(scores.max())
    keep = np.flatnonzero(scores >= min_score)
    side, top, left = geometry
    height, width = image_shape[:2]
    boxes = []
    for index in keep[np.argsort(-scores[keep])]:
        cx = reg[index, 0] / INPUT_SIZE + ANCHORS[index, 0]
        cy = reg[index, 1] / INPUT_SIZE + ANCHORS[index, 1]
        w, h = reg[index, 2] / INPUT_SIZE, reg[index, 3] / INPUT_SIZE
        box = ((cx - w / 2) * side - left, (cy - h / 2) * side - top,
               (cx + w / 2) * side - left, (cy + h / 2) * side - top)
        box = (max(0.0, float(box[0])), max(0.0, float(box[1])),
               min(float(width), float(box[2])), min(float(height), float(box[3])))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if all(iou(box, kept[:4]) < nms_iou for kept in boxes):
            boxes.append((*box, float(scores[index])))
    return boxes, best
