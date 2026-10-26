import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from modules.temporal_fusion import speech_windows, WindowJoiner


class TemporalTests(unittest.TestCase):
    def make_joiner(self):
        return WindowJoiner(speech_windows({"segments": [
            {"start": 1, "end": 2, "text": "first"},
            {"start": 2, "end": 3, "text": "second"},
        ]}, 100))

    def face(self, track=1, emotion="happy"):
        return {"face_id": track, "emotion": emotion, "confidence": 0.8}

    def test_boundary_and_silence_do_not_leak_text(self):
        joiner = self.make_joiner()
        joiner.observe(500000100, [self.face()])
        joiner.observe(1500000100, [self.face()])
        joiner.observe(2000000100, [self.face(emotion="sad")])
        first, second = list(joiner.results(3000000100))
        self.assertEqual(first["sample_count"], 1)
        self.assertEqual(first["emotion_data"]["emotion"], "happy")
        self.assertEqual(second["emotion_data"]["emotion"], "sad")

    def test_multiple_faces_block_whole_window(self):
        joiner = self.make_joiner()
        joiner.observe(1100000100, [self.face()])
        joiner.observe(1500000100, [self.face(), self.face(2)])
        event = list(joiner.results(3000000100))[0]
        self.assertEqual(event["status"], "unavailable")
        self.assertIsNone(event["emotion_data"])

    def test_track_change_and_early_stop_are_unavailable(self):
        joiner = self.make_joiner()
        joiner.observe(1100000100, [self.face()])
        joiner.observe(1500000100, [self.face(2)])
        event = list(joiner.results(1600000100))[0]
        self.assertIn("track_changed", event["uncertainty"])
        self.assertIn("incomplete_window", event["uncertainty"])

    def test_offset_and_invalid_times(self):
        window = speech_windows({"segments": [{"start": 1, "end": 2}]}, 100, -0.5)[0]
        self.assertEqual(window["start_ns"], 500000100)
        with self.assertRaises(ValueError):
            speech_windows({"segments": [{"start": 2, "end": 1}]}, 0)


class VideoPipelineTests(unittest.TestCase):
    def test_slow_inference_does_not_change_segment_alignment(self):
        from multimodal_app import MultiModalEmotionApp
        app = MultiModalEmotionApp.__new__(MultiModalEmotionApp)
        app.config = {"output": {"save_results": True}, "display": {"show_window": False}}
        app.speech_result = {"segments": [{"start": 0, "end": 1, "text": "one"},
                                             {"start": 1, "end": 2, "text": "two"}]}
        app.emotion_detector = Mock()
        app.emotion_detector.detect.side_effect = [
            [{"face_id": 1, "emotion": "happy", "confidence": 0.8}],
            [{"face_id": 1, "emotion": "sad", "confidence": 0.9}],
        ]
        app.multimodal_analyzer = Mock()
        app.multimodal_analyzer.analyze.return_value = {}
        app.visualizer = Mock()
        app.visualizer.draw.side_effect = lambda frame, *_: frame
        app._log_emotion_change = Mock()
        app._print_final_statistics = Mock()
        app.frame_count = 0
        app.prev_time = 0
        app.logger = Mock()
        capture = Mock()
        capture.read.side_effect = [(True, np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(2)] + [(False, None)]
        positions = iter([0, 1000])
        import cv2
        capture.get.side_effect = lambda prop: 1.0 if prop == cv2.CAP_PROP_FPS else next(positions)
        with tempfile.TemporaryDirectory() as output:
            app.output_dir = Path(output)
            with patch("multimodal_app.cv2.VideoCapture", return_value=capture):
                app._process_video("fixture.mp4", with_audio=True)
            events = [json.loads(line) for line in (app.output_dir / "fusion_windows.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["speech_data"]["text"] for e in events], ["one", "two"])
        self.assertEqual([e["emotion_data"]["emotion"] for e in events], ["happy", "sad"])
        self.assertEqual([e["status"] for e in events], ["aligned", "aligned"])
        self.assertEqual(app.multimodal_analyzer.analyze.call_count, 2)
        capture.release.assert_called_once()
