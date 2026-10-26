"""Timing, missing-data and evidence attacks against the opt-in contract."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from modules.av_visible_speech import CONTRACT, MAX_MEDIA_LAG_MS, apply_contract, assess, groups, machine_record
from modules.driver_observation_features import compact_observations, rule_recognition
from modules.driver_recognition import infer_recognition, normalize_model_recognition, recognition_schema, validate_recognition
from modules.driver_response_policy import run_saved_observations
from test_speech_mouth_diagnostic import THRESHOLDS, rows
from modules.speech_mouth_diagnostic import speech_mouth_diagnostic


def observations(lag=-40, index=0):
    values = rows()
    values[12]['hand_mouth']['aperture'] = 0.04
    diagnostic = speech_mouth_diagnostic(values, [0, 1], 1, THRESHOLDS)
    diagnostic.pop('source_geometry_frame_indices')
    record = {'schema': 'ara-av-correspondence-v1', 'segment_index': index, 'interval_seconds': [0, 1],
        'correspondence_status': 'in_frame_candidate', 'candidate_track_id': 7, 'geometry_track_id': 7,
        'hold_reasons': [], 'speaker_verified': False, 'driver_state_usable': False, 'screen_status': 'unknown',
        'scan': {'status': 'diagnostic_candidate', 'candidate_offset_ms': lag, 'accepted_offset_ms': None,
                 'peak_at_search_boundary': False, 'pairs': 12, 'confidence_margin': 5, 'min_distance': 8}}
    return values, diagnostic, record


def compact(lag=-40, index=0):
    _, diagnostic, record = observations(lag, index)
    features = [{'evidence_id': prefix + str(index), 'kind': kind, 'interval_seconds': [0, 1], 'value': value}
                for prefix, kind, value in [('E_asr', 'asr', {'text': 'untrusted sensor words'}),
                    ('E_mouth', 'mouth_asr_coincidence', {'samples': 24, 'aperture_range': 0.037}),
                    ('E_speech_mouth', 'speech_mouth_diagnostic', diagnostic),
                    ('E_av_sync', 'av_correspondence', {'record': machine_record(record)})]]
    return {'features': features, 'experimental_candidates': {'visible_person_speaks': ['E_asr'+str(index), 'E_mouth'+str(index)],
            'offscreen_voice': [], 'hand_over_mouth': [], 'nod': []}}


class FakeClient:
    def __init__(self, value): self.value = value
    def generate(self, *args, **kwargs): return {'status': 'generated', 'text': json.dumps(self.value)}


class AVVisibleSpeechTests(unittest.TestCase):
    def test_shifted_correspondence_cannot_be_current_media_speech(self):
        baseline = compact()
        snapshot = deepcopy(baseline)
        self.assertEqual(rule_recognition(apply_contract(baseline))['events']['visible_person_speaks']['value'], 'yes')
        for lag in (-560, -520, 440, 480, 520):
            shifted = compact(lag)
            self.assertEqual(rule_recognition(shifted)['events']['visible_person_speaks']['value'], 'yes')
            revised = apply_contract(shifted)
            self.assertEqual(rule_recognition(revised)['events']['visible_person_speaks']['value'], 'unknown')
            self.assertIn('av_candidate_outside_media_time_band', revised['features'][-1]['value']['hold_reasons'])
        self.assertEqual(baseline, snapshot)

    def test_band_boundary_and_invalid_numbers_fail_closed(self):
        _, diagnostic, record = observations()
        for lag, expected in [(-80, True), (80, True), (80.001, False), (-80.001, False),
                              (math.nan, False), (math.inf, False), (True, False), (None, False)]:
            record['scan']['candidate_offset_ms'] = lag
            with self.subTest(lag=lag):
                gate = assess(0, [0, 1], diagnostic, record)
                self.assertEqual(gate['status'] == 'near_media_time_candidate', expected)
                json.dumps(gate, allow_nan=False)
        self.assertEqual(MAX_MEDIA_LAG_MS, 80)

    def test_geometry_exclusions_and_missing_edges_do_not_become_speaker_evidence(self):
        values, _, record = observations()
        variants = [values[:12] + values[13:], values[5:], values[:-5]]
        for field, value in [('face_count', 2), ('status', 'backend_error')]:
            altered = deepcopy(values); altered[12][field] = value; variants.append(altered)
        altered = deepcopy(values); altered[12]['hand_mouth']['status'] = 'hand_overlap_risk'; variants.append(altered)
        for altered in variants:
            diagnostic = speech_mouth_diagnostic(altered, [0, 1], 1, THRESHOLDS)
            self.assertEqual(assess(0, [0, 1], diagnostic, record)['status'], 'unknown')

    def test_missing_held_boundary_or_weak_scan_never_uses_raw_lag(self):
        _, diagnostic, record = observations()
        variants = [None]
        for target, key, value in [('root','correspondence_status','unknown'), ('root','hold_reasons',['occlusion']),
                ('scan','peak_at_search_boundary',True), ('scan','status','flat_curve'), ('scan','pairs',7),
                ('scan','pairs',True), ('scan','confidence_margin',2.99), ('scan','min_distance',10.01)]:
            altered = deepcopy(record)
            (altered if target == 'root' else altered['scan'])[key] = value
            variants.append(altered)
        for altered in variants:
            gate = assess(0, [0, 1], diagnostic, altered)
            self.assertEqual(gate['status'], 'unknown')
            self.assertIsNone(gate['reliable_correspondence_lag_ms'])
            self.assertIsNone(gate['accepted_offset_ms'])

    def test_track_index_interval_and_attribution_claims_rejected(self):
        _, diagnostic, record = observations()
        for key, value in [('candidate_track_id', True), ('geometry_track_id', 8), ('segment_index', 1),
                ('segment_index', True), ('interval_seconds', [0.1, 1]), ('speaker_verified', True),
                ('driver_state_usable', True), ('screen_status', 'in_frame')]:
            altered = deepcopy(record); altered[key] = value
            self.assertEqual(assess(0, [0, 1], diagnostic, altered)['status'], 'unknown')
        record['scan']['accepted_offset_ms'] = -40
        self.assertEqual(assess(0, [0, 1], diagnostic, record)['status'], 'unknown')

    def test_low_motion_is_not_offscreen_person_or_closed_lips(self):
        _, _, record = observations()
        diagnostic = speech_mouth_diagnostic(rows(), [0, 1], 1, THRESHOLDS)
        gate = assess(0, [0, 1], diagnostic, record)
        self.assertEqual(gate['status'], 'unknown')
        self.assertFalse(gate['speaker_verified'])
        self.assertEqual(gate['speech_source'], 'unknown')
        self.assertIn('audio_playback', gate['unresolved_alternatives'])

    def test_four_same_segment_evidence_ids_are_required_even_if_features_exist(self):
        value = compact()
        value['features'] += compact(index=1)['features']
        revised = apply_contract(value)
        recognition = rule_recognition(revised)
        self.assertEqual(len(recognition['events']['visible_person_speaks']['evidence_ids']), 4)
        validate_recognition(recognition, revised)
        for ids in [['E_asr0','E_mouth0'], ['E_asr0','E_mouth0','E_av_sync1','E_visible_speech1']]:
            recognition['events']['visible_person_speaks']['evidence_ids'] = ids
            with self.assertRaisesRegex(ValueError, 'matched_asr_mouth_av_gate_required'):
                validate_recognition(recognition, revised)

    def test_forged_gate_or_candidates_cannot_override_timing_hold(self):
        revised = apply_contract(compact(500))
        valid = apply_contract(compact())
        revised['features'][-1] = deepcopy(valid['features'][-1])
        revised['experimental_candidates'] = deepcopy(valid['experimental_candidates'])
        self.assertEqual(groups(revised), [])
        recognition = rule_recognition(valid)
        with self.assertRaisesRegex(ValueError, 'matched_asr_mouth_av_gate_required'):
            validate_recognition(recognition, revised)

    def test_evidence_interval_mismatch_and_duplicates_fail_closed(self):
        for position in (1, 2, 3):
            value = compact(); value['features'][position]['interval_seconds'] = [0.2, 1]
            self.assertEqual(groups(apply_contract(value)), [])
        value = compact(); value['features'].append(deepcopy(value['features'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate_feature_id'): apply_contract(value)

    def test_schema_is_versioned_and_model_cannot_fall_back_to_v3(self):
        revised = apply_contract(compact())
        schema = recognition_schema([], revised)
        self.assertEqual(schema['properties']['schema']['const'], 'ara-recognition-model-v4')
        branches = schema['properties']['events']['properties']['visible_person_speaks']['anyOf']
        self.assertEqual(branches[1]['properties']['evidence_ids']['const'], groups(revised)[0])
        raw = {'schema': 'ara-recognition-model-v4', 'events': {
            k: {a: b for a, b in entry.items() if a != 'uncertainty'} for k, entry in rule_recognition(revised)['events'].items()}}
        self.assertEqual(infer_recognition(FakeClient(raw), revised, time.monotonic()+5)['status'], 'accepted')
        raw['schema'] = 'ara-recognition-model-v3'
        self.assertEqual(infer_recognition(FakeClient(raw), revised, time.monotonic()+5)['status'], 'invalid_output')
        self.assertEqual(infer_recognition(FakeClient([]), revised, time.monotonic()+5)['status'], 'invalid_output')

    def test_model_unknown_and_measurement_limitations_are_enforced(self):
        revised = apply_contract(compact())
        prediction = rule_recognition(revised)
        prediction['events']['visible_person_speaks']['uncertainty'] = ['uncalibrated_geometry','speaker_unverified']
        with self.assertRaisesRegex(ValueError, 'av_measurement_uncertainty_missing'):
            validate_recognition(prediction, revised)
        raw = {'schema': 'ara-recognition-model-v4', 'events': {
            name: {'value': 'unknown', 'evidence_ids': []} for name in revised['experimental_candidates']}}
        raw['events']['visible_person_speaks']['evidence_ids'] = ['E_asr0']
        with self.assertRaisesRegex(ValueError, 'unknown_evidence_schema'): normalize_model_recognition(raw)

    def test_real_compaction_maps_gate_to_raw_evidence_and_preserves_legacy(self):
        values, _, record = observations()
        registry = [{'evidence_id': f'geometry.{i}', 'relative_path': 'geometry.jsonl'} for i in range(24)]
        registry += [{'evidence_id': 'asr.0'}, {'evidence_id': 'av_sync.0'}]
        env = {'input': {'interval_seconds': [0, 1], 'common_audio_interval_seconds': [0, 1]},
            'evidence_registry': registry, 'observations': [
                {'kind':'asr','status':'observed','value':{'start':0,'end':1,'text':'machine words'},'evidence_ids':['asr.0']},
                {'kind':'av_correspondence','status':'observed','value':record,'evidence_ids':['av_sync.0']}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'envelope.json'; path.write_text('{}')
            with patch('modules.driver_observation_features.read_json', return_value=env), \
                    patch('modules.driver_observation_features.validate_envelope'), \
                    patch('modules.driver_observation_features.read_lines', return_value=values):
                legacy, _ = compact_observations(path, THRESHOLDS)
                revised, mapping = compact_observations(path, THRESHOLDS, recognition_contract=CONTRACT)
                audio, _ = compact_observations(path, THRESHOLDS, ('audio',), recognition_contract=CONTRACT)
        self.assertNotIn('recognition_contract', legacy)
        self.assertNotIn('record', next(f for f in legacy['features'] if f['kind']=='av_correspondence')['value'])
        self.assertEqual(groups(revised), [['E_asr0','E_mouth0','E_av_sync0','E_visible_speech0']])
        self.assertIn('av_sync.0', mapping['feature_evidence']['E_visible_speech0']['raw_evidence_ids'])
        self.assertEqual(groups(audio), [])

    def test_saved_app_entry_selects_contract_from_configuration(self):
        revised = apply_contract(compact())
        raw = {'schema': 'ara-recognition-model-v4', 'events': {
            k: {a: b for a, b in entry.items() if a != 'uncertainty'} for k, entry in rule_recognition(revised)['events'].items()}}
        with tempfile.TemporaryDirectory() as directory, \
                patch('modules.driver_observation_features.compact_observations', return_value=(revised, {})) as collect, \
                patch('modules.local_llm_client.LocalLLMClient', return_value=FakeClient(raw)):
            cycle = run_saved_observations('unused', directory, {'candidate_thresholds': THRESHOLDS,
                'base_url':'http://127.0.0.1:8080','model':'local','recognition_contract': CONTRACT})
            self.assertEqual(collect.call_args.kwargs['recognition_contract'], CONTRACT)
            self.assertEqual(cycle['recognition']['status'], 'accepted')
            self.assertEqual(cycle['policy']['policy'], 'wait')


if __name__ == '__main__': unittest.main()
