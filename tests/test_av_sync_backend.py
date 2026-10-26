"""Uncalibrated AV scores must not promote identity, offscreen speech or state."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from modules.av_sync_backend import (CONFIG, SCHEMA, machine_windows, validate_windows, geometry_tracks,
                                     segment_geometry, scan_embeddings, correspondence, validate_result)
from modules.observation_worker_client import SubprocessObservationClient, digest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('av_sync_worker_test', ROOT / 'multimodal-emotion-analyzer/scripts/av_sync_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def geometry():
    points = [[0.4, 0.3, 0], [0.6, 0.6, 0]] * 239
    return [{'frame_index': i, 'frame_time_s': i / 25, 'status': 'ok', 'face_count': 1,
             'observations': [{'kind': 'face_landmarks', 'landmarks': points}],
             'hand_mouth': {'status': 'no_positive_overlap', 'aperture': 0.003 + (i % 2) * 0.04}}
            for i in range(50)]


def classified():
    return [{'source_pts_seconds': i / 25, 'face_count': 1, 'faces': [{'face_id': 0}]} for i in range(50)]


class AVContractTests(unittest.TestCase):
    def test_transcript_not_passed_and_offsets_half_open(self):
        windows = machine_windows([{'start': 0, 'end': 1, 'text': 'untrusted transcript'},
                                   {'start': 1, 'end': 2, 'text': ' '}], 0.25, 3)
        self.assertEqual(windows, [{'segment_index': 0, 'interval_seconds': [0.25, 1.25]}])
        self.assertNotIn('untrusted', json.dumps(windows))
        for bad in ([{**windows[0], 'reference': 'offscreen'}], windows * 2,
                    [{'segment_index': 0, 'interval_seconds': [-1, 1]}]):
            with self.assertRaises(ValueError): validate_windows(bad, 3)

    def test_native_time_mismatch_rejected(self):
        values = classified()
        values[1]['source_pts_seconds'] += 0.1
        with self.assertRaisesRegex(ValueError, 'timeline mismatch'):
            geometry_tracks(geometry(), values)

    def test_missing_multiple_faces_and_track_switch_hold(self):
        for change in ('multiple', 'missing', 'switch', 'nonfinite'):
            with self.subTest(change=change):
                rows, classes = geometry(), classified()
                if change == 'multiple': rows[12]['face_count'] = 2
                if change == 'missing': rows[12]['observations'] = []
                if change == 'switch': classes[12]['faces'][0]['face_id'] = 1
                if change == 'nonfinite':
                    rows = deepcopy(rows)
                    rows[12]['observations'][0]['landmarks'][0][0] = float('nan')
                tracks, _ = geometry_tracks(rows, classes)
                value = segment_geometry(rows, tracks, {'segment_index': 0, 'interval_seconds': [0, 1]}, 2)
                self.assertIn('track_changed_or_ambiguous', value['hold_reasons'])

    def test_occlusion_holds_even_when_sync_scores_are_strong(self):
        rows = geometry()
        rows[12]['hand_mouth']['status'] = 'hand_overlap_risk'
        tracks, _ = geometry_tracks(rows, classified())
        window = {'segment_index': 0, 'interval_seconds': [0, 1]}
        gate = segment_geometry(rows, tracks, window, 2)
        value = correspondence(window, gate, {'status': 'diagnostic_candidate', 'min_distance': 1,
            'confidence_margin': 5, 'candidate_offset_ms': 0, 'peak_at_search_boundary': False})
        self.assertEqual(value['correspondence_status'], 'unknown')
        self.assertIn('excluded_visual_samples', value['hold_reasons'])

    def test_strong_sync_is_only_track_candidate(self):
        tracks, _ = geometry_tracks(geometry(), classified())
        window = {'segment_index': 0, 'interval_seconds': [0, 1]}
        gate = segment_geometry(geometry(), tracks, window, 2)
        value = correspondence(window, gate, {'status': 'diagnostic_candidate', 'min_distance': 1,
            'confidence_margin': 5, 'candidate_offset_ms': 120, 'peak_at_search_boundary': False})
        self.assertEqual(value['correspondence_status'], 'in_frame_candidate')
        self.assertEqual(value['candidate_track_id'], 0)
        self.assertEqual(value['speaker_link'], {'status': 'unknown', 'track_id': None})
        for key in ('speaker_verified', 'mouth_visibility_verified', 'driver_state_usable', 'absolute_sync_measured'):
            self.assertFalse(value[key])

    def test_flat_low_motion_boundary_and_bad_scores_cannot_be_promoted(self):
        window = {'segment_index': 0, 'interval_seconds': [0, 1]}
        tracks, _ = geometry_tracks(geometry(), classified())
        for change in ('flat', 'low_motion', 'boundary', 'nan', 'distance'):
            rows = geometry()
            scan = {'status': 'diagnostic_candidate', 'min_distance': 1,
                    'confidence_margin': 5, 'candidate_offset_ms': 0, 'peak_at_search_boundary': False}
            if change == 'flat': scan['status'] = 'flat_curve'
            if change == 'low_motion':
                for row in rows: row['hand_mouth']['aperture'] = 0.003
            if change == 'boundary': scan['peak_at_search_boundary'] = True
            if change == 'nan': scan['confidence_margin'] = float('nan')
            if change == 'distance': scan['min_distance'] = 12
            gate = segment_geometry(rows, tracks, window, 2)
            self.assertEqual(correspondence(window, gate, scan)['correspondence_status'], 'unknown')

    def test_result_binding_and_unverified_link_required(self):
        window = {'segment_index': 0, 'interval_seconds': [0, 1]}
        tracks, _ = geometry_tracks(geometry(), classified())
        record = correspondence(window, segment_geometry(geometry(), tracks, window, 2),
                                {'status': 'insufficient_context'})
        result = {'schema': SCHEMA, 'human_reference_used': False, 'source_hashes': {'video': 'hash'},
                  'speaker_verified': False, 'method_accuracy_validated': False, 'records': [record]}
        self.assertTrue(validate_result(result, [window], {'video': 'hash'}))
        for change in ('source', 'time', 'link', 'state', 'reference'):
            bad = deepcopy(result)
            if change == 'source': bad['source_hashes'] = {}
            if change == 'time': bad['records'][0]['interval_seconds'] = [0, 2]
            if change == 'link': bad['records'][0]['screen_status'] = 'in_frame'
            if change == 'state': bad['records'][0]['driver_state_usable'] = True
            if change == 'reference': bad['human_reference_used'] = True
            with self.assertRaises(ValueError): validate_result(bad, [window], {'video': 'hash'})


class AVScanTests(unittest.TestCase):
    def test_offset_sign_and_same_pairs_at_every_lag(self):
        rng = np.random.default_rng(42)
        video = rng.normal(size=(150, 32)).astype(np.float32)
        for shift in (-5, 0, 5):
            audio = np.roll(video, -shift, axis=0)
            scan = scan_embeddings(video, audio, list(range(10, 140)))
            self.assertEqual(scan['candidate_offset_ms'], shift * 40)
            self.assertEqual(len({p['pairs'] for p in scan['curve']}), 1)
            self.assertFalse(scan['embedding_zero_padding'])
            self.assertIsNone(scan['accepted_offset_ms'])

    def test_nonzero_audio_origin_is_applied(self):
        rng = np.random.default_rng(1)
        video = rng.normal(size=(150, 32)).astype(np.float32)
        audio = video[6:]
        scan = scan_embeddings(video, audio, range(35, 120), audio_offset_seconds=0.24)
        self.assertEqual(scan['candidate_offset_ms'], 0)
        self.assertLess(scan['max_audio_origin_quantization_ms'], 1e-9)
        scan = scan_embeddings(video, audio, range(35, 120), audio_offset_seconds=0.25)
        self.assertEqual(scan['candidate_offset_ms'], 0)
        self.assertAlmostEqual(scan['max_audio_origin_quantization_ms'], 10)

    def test_flat_and_short_context_remain_unknown(self):
        constant = np.zeros((150, 32), dtype=np.float32)
        self.assertEqual(scan_embeddings(constant, constant, range(40, 100))['status'], 'flat_curve')
        self.assertEqual(scan_embeddings(constant[:10], constant[:10], range(5))['status'], 'insufficient_context')
        self.assertEqual(scan_embeddings(constant, constant, [])['status'], 'insufficient_context')

    def test_invalid_embeddings_indices_and_origin_rejected(self):
        valid = np.zeros((100, 4), dtype=np.float32)
        for video, audio, indices, offset in [(valid * np.nan, valid, [30], 0),
                (valid, valid, [-1], 0), (valid, valid[:, :2], [30], 0),
                (valid, valid, [30], float('inf'))]:
            with self.assertRaises(ValueError): scan_embeddings(video, audio, indices, offset)


class AVWorkerTests(unittest.TestCase):
    def test_empty_machine_asr_real_ipc_skips_model_and_returns_provenance(self):
        root = ROOT / '.tmp_devai_ara'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            folder = Path(directory)
            files = {}
            for name in ('video', 'audio', 'geometry', 'classification'):
                path = folder / (name + '.bin')
                path.write_bytes(b'bound input')
                files[name] = path
            with SubprocessObservationClient([str(ROOT / 'multimodal-emotion-analyzer/.venv/Scripts/python.exe'),
                    '-B', str(ROOT / 'multimodal-emotion-analyzer/scripts/av_sync_worker.py')],
                    folder / 'ipc', backend_id='syncnet', allowed_root=ROOT) as client:
                result = client.submit(files['video'], [0, 1], timeout_s=5,
                    clip_id='test', asr_windows=[], audio_offset_seconds=0,
                    audio_input_path=str(files['audio']), audio_sha256=digest(files['audio']),
                    geometry_path=str(files['geometry']), geometry_sha256=digest(files['geometry']),
                    classification_path=str(files['classification']), classification_sha256=digest(files['classification'])).result()
            self.assertEqual(result['status'], 'ok')
            value = json.loads(Path(result['artifacts'][0]['resolved_path']).read_text())
            self.assertEqual(value['status'], 'no_machine_asr_intervals')
            self.assertEqual(value['model_calls'], {'visual_batches': 0, 'audio_batches': 0})
            self.assertNotIn('model', value)

    def test_reference_payload_and_online_mode_rejected_before_model_load(self):
        request = {'protocol': 'observation-worker-v1', 'request_id': 'test123', 'backend_id': 'syncnet',
                   'input_path': 'unused', 'input_sha256': 'unused', 'interval_seconds': [0, 1],
                   'deadline_monotonic_ns': time.monotonic_ns() + 10**9, 'mode': 'offline', 'available_until_s': 1,
                   'asr_windows': [], 'audio_offset_seconds': 0}
        for bad in ({**request, 'human_reference': 'offscreen'}, {**request, 'mode': 'online'}):
            with patch.object(worker, 'load_runtime', side_effect=AssertionError('model must not load')):
                with self.assertRaisesRegex(ValueError, 'offline SyncNet'):
                    worker.inputs(bad)


if __name__ == '__main__':
    unittest.main()
