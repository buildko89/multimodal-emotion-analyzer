"""Adversarial missing-data cases must not become inferred speaker evidence."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modules.speech_mouth_diagnostic import speech_mouth_diagnostic
from modules.driver_observation_features import compact_observations, rule_recognition
from modules.driver_recognition import recognition_schema, validate_recognition, recognition_messages

THRESHOLDS = {'mouth_aperture_range': 0.02, 'mouth_minimum_samples': 5,
              'maximum_frame_gap_seconds': 0.15,
              'hand_mouth_risk_minimum_consecutive_frames': 3,
              'head_event_maximum_seconds': 2, 'head_event_minimum_seconds': 0.3,
              'head_pitch_excursion_degrees': 6, 'head_return_degrees': 2}


def rows():
    return [{'frame_index': i, 'frame_time_s': i / 24, 'status': 'ok', 'face_count': 1,
             'hands': [], 'head_transform': None,
             'hand_mouth': {'status': 'no_positive_overlap', 'aperture': 0.003}}
            for i in range(24)]


class SpeechMouthDiagnosticTests(unittest.TestCase):
    def inspect(self, values, interval=(0, 1)):
        return speech_mouth_diagnostic(values, interval, 1, THRESHOLDS)

    def test_low_motion_is_candidate_only_even_with_complete_geometry(self):
        value = self.inspect(rows())
        self.assertEqual(value['status'], 'audio_with_low_mouth_motion_candidate')
        self.assertTrue(value['geometry_coverage_complete'])
        for field in ('speaker_verified', 'mouth_closed_verified', 'mouth_visibility_verified', 'absolute_sync_measured'):
            self.assertFalse(value[field])
        self.assertEqual(value['offscreen_voice_decision'], 'unknown')
        self.assertIn('audio_playback', value['unresolved_alternatives'])

    def test_fixed_open_mouth_is_not_claimed_closed(self):
        values = rows()
        for row in values:
            row['hand_mouth']['aperture'] = 0.2
        value = self.inspect(values)
        self.assertEqual(value['status'], 'audio_with_low_mouth_motion_candidate')
        self.assertFalse(value['mouth_closed_verified'])
        self.assertEqual(value['aperture_min'], 0.2)

    def test_large_motion_is_not_speaker_verification(self):
        values = rows()
        values[12]['hand_mouth']['aperture'] = 0.04
        value = self.inspect(values)
        self.assertEqual(value['status'], 'audio_with_mouth_motion_candidate')
        self.assertFalse(value['speaker_verified'])
        self.assertEqual(value['speech_source'], 'unknown')

    def test_occlusion_multiple_faces_failure_and_invalid_aperture_block_candidate(self):
        changes = [('mouth', 'hand_overlap_risk'), ('faces', 2), ('status', 'backend_error'),
                   ('aperture', None), ('aperture', math.nan), ('aperture', math.inf),
                   ('aperture', -0.1), ('aperture', True)]
        for kind, item in changes:
            with self.subTest(kind=kind, value=item):
                values = rows()
                row = values[12]
                if kind == 'mouth': row['hand_mouth']['status'] = item
                elif kind == 'faces': row['face_count'] = item
                elif kind == 'status': row['status'] = item
                else: row['hand_mouth']['aperture'] = item
                result = self.inspect(values)
                self.assertEqual(result['status'], 'insufficient_geometry')
                self.assertEqual(result['usable_mouth_samples'], 23)
                self.assertEqual(len(result['source_geometry_frame_indices']), 24)
                json.dumps(result, allow_nan=False)

    def test_frame_gap_and_time_gap_and_missing_edges_block_candidate(self):
        for values in (rows()[:12] + rows()[13:], rows()[5:], rows()[:-5]):
            with self.subTest(frames=len(values)):
                value = self.inspect(values)
                self.assertEqual(value['status'], 'insufficient_geometry')
                self.assertIn('visual_time_or_frame_gap', value['geometry_blockers'])
        values = rows()
        values[12]['frame_time_s'] = values[11]['frame_time_s']
        self.assertEqual(self.inspect(values)['status'], 'insufficient_geometry')

    def test_audio_only_short_and_outside_media_intervals_hold(self):
        for values, interval in [([], (0, 1)), (rows()[:4], (0, 0.16)), (rows(), (-0.1, 1)),
                                 (rows(), (0, 1.1))]:
            with self.subTest(interval=interval, frames=len(values)):
                self.assertEqual(self.inspect(values, interval)['status'], 'insufficient_geometry')

    def test_half_open_window_excludes_next_interval_motion(self):
        values = rows()
        values[12]['hand_mouth']['aperture'] = 0.5
        self.assertEqual(self.inspect(values, (0, 0.5))['status'], 'audio_with_low_mouth_motion_candidate')

    def test_threshold_equality_is_motion_not_low_motion(self):
        values = rows()
        for row in values: row['hand_mouth']['aperture'] = 0
        values[12]['hand_mouth']['aperture'] = THRESHOLDS['mouth_aperture_range']
        self.assertEqual(self.inspect(values)['status'], 'audio_with_mouth_motion_candidate')

    def test_integration_keeps_evidence_offsets_and_unknown_schema(self):
        values = rows()
        for row in values: row['frame_time_s'] += 0.5
        registry = [{'evidence_id': f'geometry.{i}', 'relative_path': 'geometry.jsonl'} for i in range(24)]
        registry.append({'evidence_id': 'asr.0'})
        env = {'input': {'interval_seconds': [0, 2], 'common_audio_interval_seconds': [0.5, 2]},
               'evidence_registry': registry, 'observations': [
                   {'kind': 'asr', 'status': 'observed',
                    'value': {'start': 0, 'end': 1, 'text': 'machine transcript'}, 'evidence_ids': ['asr.0']}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'envelope.json'
            path.write_text('{}')
            with patch('modules.driver_observation_features.read_json', return_value=env), \
                    patch('modules.driver_observation_features.validate_envelope'), \
                    patch('modules.driver_observation_features.read_lines', return_value=values):
                compact, mapping = compact_observations(path, THRESHOLDS)
                audio, _ = compact_observations(path, THRESHOLDS, ('audio',))
                empty_env = deepcopy(env)
                empty_env['observations'][0]['value']['text'] = '  '
                with patch('modules.driver_observation_features.read_json', return_value=empty_env):
                    empty, _ = compact_observations(path, THRESHOLDS)
        feature = next(f for f in compact['features'] if f['kind'] == 'speech_mouth_diagnostic')
        self.assertEqual(feature['interval_seconds'], [0.5, 1.5])
        self.assertEqual(len(mapping['feature_evidence'][feature['evidence_id']]['raw_evidence_ids']), 25)
        self.assertEqual(feature['value']['status'], 'audio_with_low_mouth_motion_candidate')
        self.assertFalse(compact['experimental_candidates']['offscreen_voice'])
        value = rule_recognition(compact)
        self.assertEqual(value['events']['offscreen_voice']['value'], 'unknown')
        self.assertIn('speaker_unverified', value['events']['offscreen_voice']['uncertainty'])
        validate_recognition(value, compact)
        entry = recognition_schema([], compact)['properties']['events']['properties']['offscreen_voice']
        self.assertEqual(entry['properties']['value']['const'], 'unknown')
        audio_feature = next(f for f in audio['features'] if f['kind'] == 'speech_mouth_diagnostic')
        self.assertEqual(audio_feature['value']['status'], 'insufficient_geometry')
        self.assertFalse(any(f['kind'] == 'speech_mouth_diagnostic' for f in empty['features']))
        self.assertIn('does not exclude replayed audio', recognition_messages(compact)[0]['content'])
        value['events']['offscreen_voice']['value'] = 'yes'
        value['events']['offscreen_voice']['evidence_ids'] = [feature['evidence_id']]
        with self.assertRaisesRegex(ValueError, 'unsupported_positive_event'):
            validate_recognition(value, compact)


if __name__ == '__main__':
    unittest.main()
