import json
from pathlib import Path
import unittest

from models.labels import normalize_emotion_label
from models.tracking import IoUTracker, intersection_over_union


ROOT = Path(__file__).resolve().parents[1]


class LabelTests(unittest.TestCase):
    def test_backend_spellings_are_normalized(self):
        self.assertEqual(normalize_emotion_label("Happy"), "happy")
        self.assertEqual(normalize_emotion_label("Sadness"), "sad")
        self.assertEqual(normalize_emotion_label("not-a-real-label"), "unknown")


class TrackerTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)

    def test_id_survives_small_motion_and_short_dropout(self):
        tracker = IoUTracker(iou_threshold=0.3, max_missed_frames=2)
        first_id = tracker.update([(10, 10, 50, 50)])[0]
        self.assertEqual(tracker.update([(12, 10, 52, 50)])[0], first_id)
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.update([(13, 11, 53, 51)])[0], first_id)

    def test_different_faces_get_different_ids(self):
        tracker = IoUTracker()
        ids = tracker.update([(0, 0, 20, 20), (100, 100, 120, 120)])
        self.assertEqual(len(set(ids)), 2)


class ManifestTests(unittest.TestCase):
    def test_destinations_are_relative_and_unique(self):
        manifest_path = ROOT / "models" / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        destinations = [spec["destination"] for spec in manifest["models"].values()]
        self.assertEqual(len(destinations), len(set(destinations)))
        for destination in destinations:
            self.assertFalse(Path(destination).is_absolute())
            self.assertNotIn("..", Path(destination).parts)


if __name__ == "__main__":
    unittest.main()

