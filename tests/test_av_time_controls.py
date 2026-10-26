"""Control construction and post-seal scoring must not hide failed or held scans."""
from array import array
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'devai/devai_ara/scripts'))
from av_time_controls import shift_pcm, recover_relative_audio_shift, score_control


def record(lag, status='in_frame_candidate', boundary=False):
    return {'correspondence_status': status,
            'scan': {'candidate_offset_ms': lag, 'peak_at_search_boundary': boundary}}


class AVTimeControlTests(unittest.TestCase):
    def test_pcm_positive_is_audio_early_and_negative_is_delay(self):
        pcm = array('h', range(10)).tobytes()
        early = shift_pcm(pcm, 3, rate=1000)
        late = shift_pcm(pcm, -3, rate=1000)
        self.assertEqual(early, pcm[6:] + bytes(6))
        self.assertEqual(late, bytes(6) + pcm[:-6])
        self.assertEqual(shift_pcm(pcm, 0, rate=1000), pcm)
        self.assertEqual(len(early), len(pcm))

    def test_shift_is_recovered_from_actual_samples_without_injected_metadata(self):
        rng = random.Random(42)
        pcm = array('h', [rng.randrange(-20000, 20000) for _ in range(5000)]).tobytes()
        for shift in (-500, 0, 500):
            altered = shift_pcm(pcm, shift, rate=1000)
            result = recover_relative_audio_shift(pcm, altered, rate=1000)
            self.assertEqual(result['relative_audio_shift_ms'], shift)

    def test_invalid_fractional_sample_or_entirely_truncated_shift_rejected(self):
        pcm = b'\x01\x00' * 10
        for shift in (float('nan'), True, 0.5, 10, -10):
            with self.assertRaises(ValueError): shift_pcm(pcm, shift, rate=1000)

    def test_relative_difference_uses_baseline_and_40ms_tolerance(self):
        result = score_control(record(-40), record(480), 500)
        self.assertEqual(result['actual_relative_offset_ms'], 520)
        self.assertEqual(result['residual_ms'], 20)
        self.assertEqual(result['assessment'], 'relative_offset_pass')
        self.assertFalse(result['speaker_truth_verified'])

    def test_wrong_sign_or_missed_shift_is_a_failure(self):
        for lag in (-540, -40):
            value = score_control(record(-40), record(lag), 500)
            self.assertEqual(value['assessment'], 'relative_offset_fail')
            self.assertFalse(value['relative_offset_within_tolerance'])

    def test_boundary_or_lost_candidate_cannot_be_counted_pass(self):
        for value in (record(480, boundary=True), record(480, status='unknown'), record(None)):
            self.assertNotEqual(score_control(record(-40), value, 500)['assessment'], 'relative_offset_pass')

    def test_held_baseline_is_separate_from_positive_stability(self):
        held = score_control(record(-1000, 'unknown', True), record(480, 'unknown'), 500)
        self.assertEqual(held['assessment'], 'hold_preserved')
        self.assertIsNone(held['relative_offset_within_tolerance'])
        changed = score_control(record(-1000, 'unknown', True), record(480), 500)
        self.assertEqual(changed['assessment'], 'hold_changed_to_candidate')


if __name__ == '__main__':
    unittest.main()
