"""Native holes, changed tracks and partial-only support contracts."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from modules.av_local_diagnostic import partition_windows, chunk_indices, local_support


def fixture():
    rows, classification = [], []
    for index in range(50):
        stamp = index / 25
        rows.append({'frame_index': index, 'frame_time_s': stamp, 'status': 'ok', 'face_count': 1,
                     'observations': [{'kind': 'face_landmarks',
                                       'landmarks': [[0.1, 0.1, 0.0], [0.8, 0.8, 0.0]] + [[0.5, 0.5, 0.0]]*476}],
                     'hand_mouth': {'status': 'no_positive_overlap', 'aperture': index/100}})
        classification.append({'source_pts_seconds': stamp, 'face_count': 1, 'faces': [{'face_id': 0}]})
    return rows, classification


def support_fixture():
    run = {'parent_segment_index': 2, 'local_index': 0, 'parent_interval_seconds': [0, 2],
           'interval_seconds': [0.5, 1.5], 'track_id': 0, 'native_frame_indices': list(range(13, 38))}
    diagnostic = {'status': 'audio_with_mouth_motion_candidate', 'geometry_coverage_complete': True,
                  'geometry_blockers': [], 'excluded_samples_by_reason': {},
                  'usable_mouth_samples': 25, 'aperture_range': 0.1}
    record = {'schema': 'ara-av-correspondence-v1', 'segment_index': 2, 'interval_seconds': [0.5, 1.5],
              'correspondence_status': 'in_frame_candidate', 'hold_reasons': [],
              'candidate_track_id': 0, 'geometry_track_id': 0, 'speaker_verified': False,
              'driver_state_usable': False, 'screen_status': 'unknown',
              'scan': {'status': 'diagnostic_candidate', 'peak_at_search_boundary': False,
                       'pairs': 8, 'confidence_margin': 3.5, 'min_distance': 7,
                       'candidate_offset_ms': 40, 'accepted_offset_ms': None}}
    return run, diagnostic, record


class LocalAVTests(unittest.TestCase):
    def partition(self, rows, classified, window=None):
        return partition_windows(rows, classified, window or [{'segment_index': 2, 'interval_seconds': [0, 2]}], 2)[0]

    def test_native_holes_are_excluded_and_never_joined(self):
        rows, classified = fixture()
        for index in (15, 16):
            classified[index].update(face_count=0, faces=[])
        parent = self.partition(rows, classified)
        self.assertEqual([r['native_frame_indices'] for r in parent['local_runs']],
                         [list(range(15)), list(range(17, 50))])
        self.assertEqual([r['frame_index'] for r in parent['excluded_samples']], [15, 16])
        self.assertEqual(parent['interpolated_samples'], 0)
        self.assertEqual(parent['parent_speech_decision'], 'unknown')

    def test_track_changes_split_even_when_all_faces_present(self):
        rows, classified = fixture()
        for row in classified[25:]:
            row['faces'][0]['face_id'] = 8
        parent = self.partition(rows, classified)
        self.assertEqual([r['track_id'] for r in parent['local_runs']], [0, 8])
        self.assertEqual(parent['exclusion_counts'], {})

    def test_multiple_faces_overlap_and_invalid_aperture_stay_excluded(self):
        rows, classified = fixture()
        classified[10].update(face_count=2, faces=[{'face_id': 0}, {'face_id': 1}])
        rows[11]['hand_mouth']['status'] = 'hand_overlap_risk'
        rows[12]['hand_mouth']['aperture'] = float('nan')
        parent = self.partition(rows, classified)
        self.assertEqual(parent['exclusion_counts'], {'track_or_face_unavailable': 1,
                         'projected_hand_mouth_overlap': 1, 'invalid_mouth_aperture': 1})

    def test_long_native_pts_gap_splits(self):
        rows, classified = fixture()
        for index in range(25, 50):
            rows[index]['frame_time_s'] += 0.2
            classified[index]['source_pts_seconds'] += 0.2
        parent = partition_windows(rows, classified, [{'segment_index': 0, 'interval_seconds': [0, 2.2]}], 2.2)[0]
        self.assertEqual(len(parent['local_runs']), 2)

    def test_parent_boundaries_and_short_runs_retained(self):
        rows, classified = fixture()
        parent = self.partition(rows, classified, [{'segment_index': 3, 'interval_seconds': [0.51, 0.57]}])
        self.assertEqual(parent['local_runs'][0]['native_frame_indices'], [13, 14])
        self.assertEqual(parent['local_runs'][0]['interval_seconds'], [0.52, 0.57])

    def test_no_usable_geometry_returns_parent_without_runs(self):
        rows, classified = fixture()
        for row in rows:
            row['status'] = 'unavailable'
        parent = self.partition(rows, classified)
        self.assertEqual(parent['local_runs'], [])
        self.assertEqual(len(parent['excluded_samples']), 50)

    def test_invalid_native_timestamp_index_or_reference_window_rejected(self):
        for corruption in ('nan', 'order', 'index', 'reference'):
            rows, classified = fixture()
            window = [{'segment_index': 2, 'interval_seconds': [0, 2]}]
            if corruption == 'nan': rows[1]['frame_time_s'] = float('nan')
            if corruption == 'order': rows[1]['frame_time_s'] = 0
            if corruption == 'index': rows[1]['frame_index'] = True
            if corruption == 'reference': window[0]['expected_answer'] = 'yes'
            with self.assertRaises(ValueError): self.partition(rows, classified, window)

    def test_five_frame_chunk_cannot_cross_a_native_hole(self):
        rows, classified = fixture()
        classified[15].update(face_count=0, faces=[])
        run = self.partition(rows, classified)['local_runs'][0]
        # Even if a cache wrongly calls a chunk spanning the hole eligible,
        # the original run membership prevents it from being scanned.
        chosen = chunk_indices(run, list(range(45)), list(range(50)), [0]*50, rows)
        self.assertEqual(chosen, list(range(11)))

    def test_invalid_cached_chunk_rejected(self):
        rows, classified = fixture()
        run = self.partition(rows, classified)['local_runs'][0]
        for index in (-1, True, 100):
            with self.assertRaises(ValueError): chunk_indices(run, [index], list(range(50)), [0]*50, rows)

    def test_positive_support_is_never_a_whole_asr_or_current_frame_answer(self):
        run, diagnostic, record = support_fixture()
        value = local_support(run, diagnostic, record)
        self.assertEqual(value['status'], 'local_av_support_candidate')
        self.assertFalse(value['whole_parent_supported'])
        self.assertEqual(value['visible_person_speaks'], 'unknown')
        self.assertEqual(value['current_frame_speech'], 'unknown')
        self.assertFalse(value['speaker_verified'])
        self.assertIsNone(value['accepted_offset_ms'])

    def test_original_80ms_and_score_gates_held(self):
        for changes in ({'candidate_offset_ms': 120}, {'confidence_margin': 2.9},
                        {'min_distance': 10.1}, {'pairs': 7}, {'peak_at_search_boundary': True},
                        {'accepted_offset_ms': 0}):
            run, diagnostic, record = support_fixture()
            record['scan'].update(changes)
            self.assertEqual(local_support(run, diagnostic, record)['status'], 'unknown')

    def test_parent_index_interval_track_and_unsupported_attribution_rejected(self):
        for key, value in (('segment_index', 3), ('interval_seconds', [0, 2]),
                           ('geometry_track_id', 9), ('speaker_verified', True), ('driver_state_usable', True)):
            run, diagnostic, record = support_fixture()
            record[key] = value
            self.assertEqual(local_support(run, diagnostic, record)['status'], 'unknown')
        run, diagnostic, record = support_fixture()
        run['interval_seconds'] = [0.5, 2.1]
        with self.assertRaises(ValueError): local_support(run, diagnostic, record)

    def test_predictor_reference_or_control_truth_rejected_before_input_access(self):
        path = Path(__file__).resolve().parents[2] / 'devai/devai_ara/scripts/av_local_predictor.py'
        spec = importlib.util.spec_from_file_location('local_predictor_test', path)
        predictor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor)
        with patch.object(predictor, 'bound', side_effect=AssertionError('must reject before touching inputs')):
            for forbidden in ('human_reference', 'generation_prompt', 'expected_shift_ms'):
                with self.assertRaises(ValueError): predictor.validate_jobs([{forbidden: 'private'}])

    def test_cached_reference_payload_rejected_before_input_access(self):
        path = Path(__file__).resolve().parents[2] / 'devai/devai_ara/scripts/av_local_predictor.py'
        spec = importlib.util.spec_from_file_location('local_predictor_cache_test', path)
        predictor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor)
        job = {key: None for key in predictor.FIELDS}
        job['cache'] = {key: None for key in predictor.CACHE_FIELDS}
        job['cache']['expected_answer'] = 'yes'
        with patch.object(predictor, 'bound', side_effect=AssertionError('must reject before touching inputs')):
            with self.assertRaises(ValueError): predictor.validate_jobs([job])


if __name__ == '__main__':
    unittest.main()
