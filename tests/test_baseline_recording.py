"""Regression checks for reproducible app-level ARA-01 recording."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from multimodal_app import MultiModalEmotionApp
from models.detector import EmotionDetector
from models.factory import create_emotion_detector


class ConfigurationRegressionTests(unittest.TestCase):
    def test_factory_honors_mediapipe_and_other_backend_selection(self):
        for backend, target in [('mediapipe', 'emotion_detector_mediapipe.MediaPipeEmotionDetector'),
                                ('haarcascade', 'models.factory.EmotionDetector')]:
            config = {'face_detection': {'model': backend}}
            with self.subTest(backend=backend), patch(target) as constructor:
                self.assertIs(create_emotion_detector(config), constructor.return_value)
                constructor.assert_called_once_with(config)

    def test_coco_yolov8_cannot_be_used_as_face_detector(self):
        detector = EmotionDetector.__new__(EmotionDetector)
        detector.config = {'face_detection': {'model': 'yolov8'}}
        with self.assertRaisesRegex(ValueError, 'not a face detector'):
            detector._init_face_detector()


class BaselineRecordingTests(unittest.TestCase):
    def app(self, directory):
        app = MultiModalEmotionApp.__new__(MultiModalEmotionApp)
        app.config = {'output': {'save_results': True, 'save_frame_observations': True},
                      'display': {'show_window': False}}
        app.output_dir = Path(directory)
        app.speech_result = None
        app.emotion_detector = Mock()
        app.visualizer = Mock()
        app.visualizer.draw.side_effect = lambda frame, *_: frame
        app._log_emotion_change = Mock()
        app.frame_count = 0
        app.prev_time = 0
        app.logger = Mock()
        return app

    def capture(self):
        cap = Mock()
        cap.read.side_effect = [(True, np.zeros((8, 8, 3), dtype=np.uint8))] * 2 + [(False, None)]
        positions = iter([0.0, 500.0])
        cap.get.side_effect = lambda prop: 2.0 if prop == cv2.CAP_PROP_FPS else next(positions)
        return cap

    def test_no_face_and_multiple_faces_remain_in_frame_log(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.app(directory)
            faces = [{'face_id': n, 'scores': np.array([0.2, 0.8], dtype=np.float32)} for n in (1, 2)]
            app.emotion_detector.detect.side_effect = [[], faces]
            with patch('multimodal_app.cv2.VideoCapture', return_value=self.capture()):
                app._process_video('fixture.mp4')
            rows = [json.loads(row) for row in (app.output_dir / 'frame_observations.jsonl').read_text().splitlines()]
            self.assertEqual([r['face_count'] for r in rows], [0, 2])
            self.assertEqual([r['source_pts_seconds'] for r in rows], [0.0, 0.5])
            self.assertEqual([r['frame_index'] for r in rows], [0, 1])
            self.assertEqual(len(rows[1]['faces'][1]['scores']), 2)
            self.assertEqual(rows[1]['media_aligned_monotonic_ns'] - rows[0]['media_aligned_monotonic_ns'], 500000000)
            summary = json.loads((app.output_dir / 'video_summary.json').read_text())
            self.assertEqual(summary['frames_processed'], 2)
            self.assertEqual(summary['completion'], 'end_of_stream')

    def test_inference_failure_is_not_recorded_as_completed_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.app(directory)
            app.emotion_detector.detect.side_effect = RuntimeError('backend failed')
            with patch('multimodal_app.cv2.VideoCapture', return_value=self.capture()):
                with self.assertRaisesRegex(RuntimeError, 'backend failed'):
                    app._process_video('fixture.mp4')
            summary = json.loads((app.output_dir / 'video_summary.json').read_text())
            self.assertEqual(summary['completion'], 'failed')
            self.assertEqual(summary['frames_processed'], 0)

    def test_audio_config_options_and_actual_asr_result_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.app(directory)
            app.config['whisper'] = {'word_timestamps': False, 'task': 'translate'}
            recognizer = Mock()
            recognizer.transcribe_file.return_value = {'text': 'machine result', 'segments': []}
            recognizer.analyze_speech_features.return_value = {'speech_rate': 0, 'total_words': 0}
            app._ensure_whisper = Mock(return_value=recognizer)
            app._process_video = Mock()
            app.process_with_audio_file('fixture.mp4', 'actual.wav')
            recognizer.transcribe_file.assert_called_once_with('actual.wav', word_timestamps=False, task='translate')
            saved = json.loads((app.output_dir / 'speech_result.json').read_text())
            self.assertEqual(saved['result']['text'], 'machine result')
            self.assertEqual(saved['source_audio'], 'actual.wav')

    def test_sensevoice_without_word_times_is_unavailable_not_zero_and_skips_whisper(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.app(directory)
            app.config['observation_integration'] = {'enabled': True, 'asr_backend': 'sensevoice'}
            app._ensure_whisper = Mock()
            app._process_video = Mock()
            with patch('modules.observation_integration.ObservationIntegrationSession') as factory:
                factory.return_value.common_audio = 'shared.wav'
                factory.return_value.sensevoice_transcript.return_value = {'text': 'regression', 'segments': []}
                app.process_with_audio_file('fixture.mp4', 'actual.wav')
            app._ensure_whisper.assert_not_called()
            saved = json.loads((app.output_dir / 'speech_result.json').read_text())
            self.assertIsNone(saved['features']['speech_rate'])
            self.assertIsNone(saved['features']['total_words'])
            self.assertEqual(saved['features']['status'], 'unavailable')
            self.assertEqual(saved['source_audio'], 'shared.wav')


if __name__ == '__main__':
    unittest.main()
