import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from models.labels import canonical_score_map
from modules.multimodal_analyzer import MultiModalAnalyzer
from modules.temporal_fusion import WindowJoiner, speech_windows
from modules.text_sentiment import TextSentimentAnalyzer


class ObservationTests(unittest.TestCase):
    def test_aliases_compare_and_result_contains_no_psychological_claims(self):
        result = MultiModalAnalyzer().analyze({'emotion': 'Happiness', 'confidence': 0.8}, {'text': '悲しいです'})
        self.assertEqual(result['status'], 'compared')
        self.assertEqual(len(result['signal_mismatches']), 1)
        self.assertEqual(result['emotion_data']['emotion'], 'happy')
        encoded = json.dumps(result, allow_nan=False)
        for key in ('trust_score', 'stress_score', 'fake_smile', 'recommendations', 'overall_state'):
            self.assertNotIn(key, encoded)
        self.assertEqual(result['observations']['acoustic_emotion'], 'not_implemented')
        self.assertEqual(result['observations']['face']['input_quality_status'], 'not_measured')

    def test_unknown_low_and_invalid_confidence_cannot_produce_mismatch(self):
        for face in ({'emotion': 'Contempt', 'confidence': 0.9},
                     {'emotion': 'Happy', 'confidence': 0.1},
                     {'emotion': 'Happy', 'confidence': float('nan')},
                     {'emotion': 'Happy', 'confidence': 2},
                     {'emotion': 'Happy', 'confidence': 0.8, 'quality': {'status': 'unavailable'}}):
            with self.subTest(face=face):
                result = MultiModalAnalyzer().analyze(face, {'text': '悲しいです'})
                self.assertEqual(result['status'], 'unavailable')
                self.assertEqual(result['signal_mismatches'], [])
                json.dumps(result, allow_nan=False)

    def test_missing_text_evidence_is_unknown(self):
        analyzer = MultiModalAnalyzer()
        for speech in (None, {'text': ''}, {'text': '机があります'}):
            result = analyzer.analyze({'emotion': 'happy', 'confidence': 0.9}, speech)
            self.assertEqual(result['status'], 'unavailable')
            self.assertEqual(result['observations']['text_polarity'], 'UNKNOWN')
        self.assertEqual(analyzer.get_history_summary()['unavailable_count'], 3)

    def test_class_order_and_unknown_probability_are_preserved(self):
        result = canonical_score_map(['Neutral', 'Happiness', 'Contempt'], [0.2, 0.3, 0.5])
        self.assertEqual(result['neutral'], 0.2)
        self.assertEqual(result['happy'], 0.3)
        self.assertEqual(result['unknown'], 0.5)
        with self.assertRaises(ValueError):
            canonical_score_map(['Neutral'], [0.2, 0.8])

    def test_text_reports_counts_and_does_not_infer_mental_state(self):
        result = TextSentimentAnalyzer().analyze_comprehensive('えっと、あの、不安です。なぜですか？')
        self.assertGreater(result['fillers']['count'], 0)
        self.assertEqual(result['sentiment']['score_kind'], 'keyword_ratio_not_probability')
        self.assertNotIn('overall_state', result)
        self.assertNotIn('stress_score', json.dumps(result))

    def test_smoothing_cannot_hide_unknown_or_failed_inference(self):
        from models.detector import EmotionDetector
        detector = EmotionDetector.__new__(EmotionDetector)
        detector.smoothing_enabled = True
        detector.smoothing_window = 6
        detector.min_confidence = 0.3
        detector.emotion_history = {}
        detector._smooth_emotion(1, 'happy', 0.9)
        self.assertEqual(detector._smooth_emotion(1, 'unknown', 0.8)[0], 'unknown')
        self.assertNotIn(1, detector.emotion_history)
        detector._smooth_emotion(1, 'happy', 0.9)
        self.assertEqual(detector._smooth_emotion(1, 'neutral', 0)[0], 'unknown')

    def test_hsemotion_reads_rgb_class_probabilities_without_auxiliary_outputs(self):
        from models.detector import EmotionDetector
        detector = EmotionDetector.__new__(EmotionDetector)
        detector.emotion_labels = ['Anger', 'Happiness']
        detector.emotion_model = Mock()
        detector.emotion_model.predict_emotions.return_value = ('Happiness', np.array([0.2, 0.8, 4.0, 5.0]))
        image = np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8)
        label, confidence, scores = detector._recognize_emotion_hsemotion(image)
        self.assertEqual(confidence, 0.8)
        self.assertEqual(len(scores), 2)
        np.testing.assert_array_equal(detector.emotion_model.predict_emotions.call_args.args[0][0, 0], [30, 20, 10])


class WindowQualityTests(unittest.TestCase):
    def test_one_unknown_sample_blocks_even_when_majority_is_valid(self):
        joiner = WindowJoiner(speech_windows({'segments': [{'start': 0, 'end': 2}]}, 0))
        for time_ns, label in ((0, 'Happy'), (500000000, 'happy'), (1000000000, 'Contempt')):
            joiner.observe(time_ns, [{'face_id': 1, 'emotion': label, 'confidence': 0.8}])
        result = list(joiner.results(2000000000))[0]
        self.assertEqual(result['status'], 'unavailable')
        self.assertIn('unknown_label', result['uncertainty'])

    def test_low_confidence_and_ties_block(self):
        for second in ({'emotion': 'happy', 'confidence': 0.1}, {'emotion': 'sad', 'confidence': 0.9}):
            joiner = WindowJoiner(speech_windows({'segments': [{'start': 0, 'end': 2}]}, 0))
            joiner.observe(0, [{'face_id': 1, 'emotion': 'happy', 'confidence': 0.8}])
            joiner.observe(1000000000, [dict(second, face_id=1)])
            self.assertEqual(list(joiner.results(2000000000))[0]['status'], 'unavailable')


class OutputTests(unittest.TestCase):
    def test_csv_and_overlay_use_observation_schema(self):
        from multimodal_app import MultiModalEmotionApp
        app = MultiModalEmotionApp.__new__(MultiModalEmotionApp)
        app.config = {'output': {'save_results': True}}
        app.prev_emotion_state = None
        face = {'emotion': 'Happy', 'confidence': 0.8, 'face_id': 1, 'monotonic_ns': 100}
        analyzer = MultiModalAnalyzer()
        with tempfile.TemporaryDirectory() as output:
            app.emotion_log_file = Path(output) / 'observation_timeline_v2.csv'
            app._init_emotion_log()
            app._log_emotion_change(face, analyzer._emotion_only_state(face), False)
            with app.emotion_log_file.open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['Face_Label'], 'happy')
            self.assertEqual(rows[0]['Input_Quality_Status'], 'not_measured')
            self.assertNotIn('Stress_Score', rows[0])
        with patch('multimodal_app.cv2.putText', side_effect=lambda frame, *_: frame) as draw:
            app._draw_multimodal_results(np.zeros((240, 640, 3), dtype=np.uint8), analyzer.analyze(face))
        labels = ' '.join(call.args[1] for call in draw.call_args_list)
        self.assertIn('Model confidence', labels)
        self.assertNotIn('Trust:', labels)
        self.assertNotIn('Stress:', labels)
