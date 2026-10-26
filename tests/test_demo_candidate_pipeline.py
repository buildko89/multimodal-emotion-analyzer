"""Regression of evidence overflow, candidate isolation and response boundaries."""
from copy import deepcopy
import json
import unittest

from modules.demo_candidate_pipeline import collect_candidates, supported_rule_recognition, response_cycle
from modules.driver_observation_features import rule_recognition
from modules.driver_recognition import validate_recognition
from modules.observation_integration import QUESTIONS


def compact(segment_count=3):
    features = [dict(evidence_id=f'{prefix}{i}', kind='asr' if prefix=='E_asr' else 'mouth_asr_coincidence', value={})
                for i in range(segment_count) for prefix in ('E_asr','E_mouth')]
    features.append(dict(evidence_id='E_head', kind='head_pitch_geometry', value={}))
    return dict(features=features, experimental_candidates={q:
        ([f'{p}{i}' for i in range(segment_count) for p in ('E_asr','E_mouth')] if q=='visible_person_speaks'
         else ['E_head'] if q=='nod' else []) for q in QUESTIONS}, limitations=[], human_reference_used=False)


class Client:
    def __init__(self, bad=False): self.calls=0; self.bad=bad
    def generate(self, messages, schema, **kwargs):
        self.calls+=1
        payload=json.loads(messages[-1]['content'])
        if self.bad: payload['text']='あなたは眠いです。'
        return dict(status='generated',text=json.dumps(payload))


class DemoPipelineTests(unittest.TestCase):
    def test_multiple_asr_segments_reproduce_old_failure_and_use_complete_pair(self):
        for segments in (3,4):
            with self.subTest(asr_segments=segments):
                value=compact(segments); before=deepcopy(value)
                with self.assertRaisesRegex(ValueError,'invented_evidence'):
                    validate_recognition(rule_recognition(value),value)
                result=supported_rule_recognition(value)
                self.assertEqual(result['events']['visible_person_speaks']['evidence_ids'],['E_asr0','E_mouth0'])
                validate_recognition(result,value)
                self.assertEqual(value,before)

    def test_unregistered_candidate_is_rejected(self):
        value=compact(); value['experimental_candidates']['nod']=['E_forged']
        with self.assertRaisesRegex(ValueError,'invented_evidence'):
            supported_rule_recognition(value)

    def test_candidate_failure_is_isolated(self):
        def build(path,thresholds,modalities,recognition_contract):
            if modalities==('audio',): raise ValueError('audio_fixture_failure')
            value=compact(); value['experimental_candidates']['visible_person_speaks']=[]
            return value,{}
        result=collect_candidates('unused',{},builder=build)['candidates']
        self.assertEqual(result['C1_audio']['status'],'inference_failure')
        self.assertEqual(sum(v['status']=='accepted_candidate' for v in result.values()),4)

    def test_response_llm_does_not_erase_rule_candidate(self):
        client=Client()
        result=response_cycle(compact(),client=client,
            preferences=[dict(source='explicit_user',preference='minimal_guidance')],guidance='必要な案内だけ表示します。')
        self.assertEqual(client.calls,1)
        self.assertEqual(result['recognition']['accepted']['events']['nod']['value'],'yes')
        self.assertEqual(result['response']['status'],'accepted')
        self.assertTrue(all(e['value']=='unknown' for e in result['recognition']['accepted']['states'].values()))

    def test_no_conversation_does_not_call_llm(self):
        client=Client()
        result=response_cycle(compact(),client=client,preferences=[dict(source='explicit_user',preference='no_conversation')])
        self.assertEqual(client.calls,0)
        self.assertEqual(result['response']['accepted']['text'],'')

    def test_asr_cannot_set_user_preference(self):
        with self.assertRaises(ValueError):
            response_cycle(compact(),preferences=[dict(source='asr',preference='rest_information')])

    def test_unsupported_llm_text_falls_back_and_observations_remain(self):
        result=response_cycle(compact(),client=Client(bad=True),
            preferences=[dict(source='explicit_user',preference='rest_information')])
        self.assertEqual(result['response']['status'],'fallback')
        self.assertNotIn('眠い',result['response']['accepted']['text'])
        self.assertEqual(result['recognition']['accepted']['events']['nod']['value'],'yes')

    def test_missing_candidates_do_not_become_negative_answers(self):
        value=compact(); value['experimental_candidates']={q:[] for q in QUESTIONS}
        result=response_cycle(value)
        self.assertTrue(all(e['value']=='unknown' for e in result['recognition']['accepted']['events'].values()))
        self.assertEqual(result['policy']['policy'],'wait')


if __name__=='__main__': unittest.main()
