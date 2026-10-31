"""VIDEO-mode eye geometry and raw blendshapes; no calibrated state detector."""
import importlib.metadata
import json
import math
from pathlib import Path
import time

import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def eye_ratio(points, indices, width, height):
    p = np.asarray([[points[i].x*width, points[i].y*height] for i in indices])
    horizontal = np.linalg.norm(p[0]-p[3])
    if horizontal <= 1e-6:
        return None
    return float((np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2*horizontal))


class FaceLandmarkerAdapter:
    def __init__(self):
        import hashlib
        model = ROOT / "runtime" / "face_landmarker" / "model" / "face_landmarker.task"
        manifest = json.loads((ROOT / "additional_model_manifest.json").read_text(encoding="utf-8"))
        record = next(e for e in manifest["files"] if e["model"] == "MediaPipe Face Landmarker")
        if hashlib.sha256(model.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError("Face bundle integrity mismatch")
        if importlib.metadata.version("mediapipe") != "1.0.1":
            raise RuntimeError("Unexpected MediaPipe version")
        self.provenance = {"model_id": "MediaPipe Face Landmarker", "bundle_version": "1",
                           "weight_sha256": record["sha256"], "runtime_version": "1.0.1", "device": "cpu"}
        self.options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=2,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(self.options)
        self.last_timestamp_ms = None

    def reset_graph(self):
        """Fresh VIDEO graph for a new recording, without re-verifying the bundle.

        On the device, a graph reused across 4 s windows lost the face for the
        first ~2 s of a window while the frame-loop BlazeFace (CPU) saw it in every frame.
        """
        self.detector.close()
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(self.options)
        self.last_timestamp_ms = None

    def close(self):
        self.detector.close()

    def analyze(self, rgb, *, asset_id, timestamp_s):
        received = time.perf_counter()
        result = {"schema": "driver-observation-v0.1", "asset_id": asset_id, "mode": "offline",
                  "frame_time_s": timestamp_s if isinstance(timestamp_s, (int, float)) and math.isfinite(timestamp_s) else None,
                  "status": "not_run", "observations": [], "state_candidates": [], "response": None,
                  "eye_geometry": None, "head_transform": None, "model": self.provenance,
                  "uncertainty": ["state_not_evaluated", "driver_identity_unverified"], "timing": {"received_monotonic_s": received}}
        def finish(status, reason=None):
            result["status"] = status
            if reason:
                result["uncertainty"].append(reason)
            done = time.perf_counter()
            result["timing"].update(completed_monotonic_s=done, processing_s=done-received)
            return result
        if rgb is None:
            return finish("unavailable", "video_unavailable")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 32:
            return finish("invalid_input", "expected_rgb_uint8")
        if result["frame_time_s"] is None or timestamp_s < 0:
            return finish("invalid_input", "invalid_frame_time")
        ms = round(timestamp_s*1000)
        if self.last_timestamp_ms is not None and ms <= self.last_timestamp_ms:
            return finish("invalid_input", "nonincreasing_quantized_frame_time")
        result["timestamp_ms"] = ms
        self.last_timestamp_ms = ms
        try:
            raw = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ms)
            result["face_count"] = len(raw.face_landmarks)
            if not raw.face_landmarks:
                return finish("unavailable", "face_unavailable")
            if len(raw.face_landmarks) != 1:
                return finish("unavailable", "multiple_people")
            landmarks = raw.face_landmarks[0]
            xyz = [[p.x, p.y, p.z] for p in landmarks]
            blendshapes = [{"index": p.index, "label": p.category_name, "score": p.score} for p in raw.face_blendshapes[0]]
            matrix = np.asarray(raw.facial_transformation_matrixes[0])
            if len(xyz) != 478 or len(blendshapes) != 52 or matrix.shape != (4,4) or not np.isfinite(xyz).all() or not np.isfinite(matrix).all() or not all(math.isfinite(p["score"]) for p in blendshapes):
                raise RuntimeError("Unexpected or nonfinite face output")
            height, width = rgb.shape[:2]
            result["eye_geometry"] = {
                "left_ratio": eye_ratio(landmarks, [362,385,387,263,373,380], width, height),
                "right_ratio": eye_ratio(landmarks, [33,160,158,133,153,144], width, height),
                "eye_closed": None, "calibrated": False}
            result["head_transform"] = matrix.tolist()
            result["observations"] = [{"kind": "face_landmarks", "landmarks": xyz, "blendshapes": blendshapes,
                                       "shape": [478,3], "blendshape_count": 52, "finite": True}]
            return finish("ok")
        except Exception as error:
            result["error"] = {"type": type(error).__name__, "message": str(error)}
            return finish("backend_error", "face_execution_failed")
