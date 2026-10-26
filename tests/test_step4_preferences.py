"""Boundary checks for explicit changes, cancellation and display provenance."""
from copy import deepcopy
import unittest

from modules.driver_response_policy import resolve_preferences, select_policy
from modules.framebuffer_display import asr_display_lines, visible_response


class Step4Tests(unittest.TestCase):
    def test_clear_keeps_history_and_does_not_restore_older_preference(self):
        history = [{'source':'explicit_user','preference':'minimal_guidance'},
                   {'source':'explicit_user','preference':'no_conversation'},
                   {'source':'explicit_user','action':'clear'}]
        original = deepcopy(history)
        self.assertIsNone(resolve_preferences(history)['active'])
        self.assertEqual(select_policy({}, history, '以前の案内')['text'], '')
        self.assertEqual(history, original)
        self.assertEqual(len(resolve_preferences(history)['history']), 3)

    def test_asr_cannot_set_or_clear_or_claim_explicit_promotion(self):
        for event in ({'source':'asr','preference':'no_conversation'},
                      {'source':'asr','action':'clear'},
                      {'source':'explicit_user','preference':'no_conversation','automatic_from_asr':True},
                      {'source':'explicit_user','action':'clear','preference':'minimal_guidance'},
                      {'source':'explicit_user','action':'erase'}):
            with self.assertRaises(ValueError): resolve_preferences([event])

    def test_empty_asr_is_not_proof_of_silence_and_correction_is_separate(self):
        self.assertIn('無発話とは断定', ''.join(asr_display_lines({'asr_text':''}, 10)))
        snapshot = {'asr_text':'会話をやめて', 'asr_source_end_ns':10,
                    'asr_correction':{'source':'explicit_user','text':'画面を見せて'}}
        original = deepcopy(snapshot)
        lines = ''.join(asr_display_lines(snapshot, 20))
        self.assertIn('認識：会話をやめて', lines)
        self.assertIn('本人訂正（別記）：画面を見せて', lines)
        self.assertEqual(snapshot, original)
        self.assertIn('希望へ自動反映なし', lines)
        snapshot['asr_correction']['source'] = 'asr'
        with self.assertRaises(ValueError): asr_display_lines(snapshot, 20)

    def test_old_and_future_asr_and_previous_policy_response(self):
        lines = asr_display_lines({'asr_text':'以前の認識','asr_source_end_ns':1}, 301*10**9)
        self.assertIn('期限切れ', ''.join(lines))
        self.assertNotIn('以前の認識', ''.join(lines))
        with self.assertRaises(ValueError):
            asr_display_lines({'asr_text':'未来','asr_source_end_ns':30}, 20)
        result = {'source_end_ns':10, 'text':'古い希望の案内','preference_revision':1}
        self.assertNotIn(result['text'], visible_response(result, 20, 300, 2)[0])


if __name__ == '__main__': unittest.main()
