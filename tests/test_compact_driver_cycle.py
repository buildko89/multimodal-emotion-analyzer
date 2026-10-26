"""Guard compact transport without weakening evidence or response checks."""
from copy import deepcopy
import json
import time
import unittest

from modules.compact_driver_cycle import decision_options, decode_decisions, decode_code, code_choices, project_candidate_observations, warmup_compact_cycle
from modules.driver_response_policy import DriverCooperativePipeline
from tests.test_driver_cooperative_ai import COMPACT


class Client:
    def __init__(self, value=None, *, failure=False, sleep=0):
        self.value, self.failure, self.sleep, self.calls = value, failure, sleep, []
    def generate(self, messages, schema, **kwargs):
        self.calls.append((messages,schema,kwargs))
        time.sleep(self.sleep)
        return {'status':'inference_failure','reason':'timeout'} if self.failure else {
            'status':'generated','text':json.dumps(self.value,ensure_ascii=False)}


class CompactCycleTests(unittest.TestCase):
    def call(self, client, **kwargs):
        return DriverCooperativePipeline(client).process(COMPACT, transport='compact-cycle-v1', **kwargs)

    def test_one_call_normalizes_candidate_and_copies_actual_generated_text(self):
        text = '休憩の際は、安全な場所に停車してから休みましょう。'
        client = Client({'d':[0,0,0,1],'t':text})
        result = self.call(client, preferences=[{'source':'explicit_user','preference':'rest_information'}])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result['recognition']['status'],'accepted')
        self.assertEqual(result['recognition']['accepted']['events']['nod']['evidence_ids'],['E_head'])
        self.assertEqual(result['response']['status'],'accepted')
        self.assertTrue(result['response']['generated_text_adopted'])
        self.assertTrue(all(v['value']=='unknown' for v in result['recognition']['accepted']['states'].values()))

    def test_each_question_index_is_checked_even_when_global_schema_allows_it(self):
        for decisions in ([0,1,0,0],[0,0,0,True],[0,0,0,-1],[0,0,0,2],[0,0,0],[0,0,0,'1']):
            with self.subTest(decisions=decisions):
                result = self.call(Client({'d':decisions,'t':''}))
                self.assertEqual(result['recognition']['status'],'invalid_output')
                self.assertIsNone(result['recognition']['accepted'])

    def test_unsupported_diagnosis_cannot_be_adopted(self):
        result = self.call(Client({'d':[0,0,0,1],'t':'あなたは眠いです。'}),
            preferences=[{'source':'explicit_user','preference':'rest_information'}])
        self.assertEqual(result['response']['status'],'fallback')
        self.assertFalse(result['response']['generated_text_adopted'])
        self.assertIsNone(result['recognition']['accepted'])

    def test_timeout_and_late_generation_fallback(self):
        for client in (Client(failure=True), Client({'d':[0,0,0,0],'t':''},sleep=.03)):
            result = self.call(client, timeout_s=.01)
            self.assertEqual(result['recognition']['status'],'inference_failure')
            self.assertIsNone(result['recognition']['accepted'])

    def test_explicit_no_conversation_is_preserved(self):
        client = Client({'d':[0,0,0,0],'t':''})
        result = self.call(client, preferences=[{'source':'explicit_user','preference':'no_conversation'}])
        self.assertEqual(result['policy']['policy'],'reduce_dialogue')
        self.assertEqual(result['response']['accepted']['text'],'')
        self.assertEqual(client.calls[0][1]['properties']['t']['const'],'')

    def test_correction_preserves_raw_machine_value_and_source(self):
        result = self.call(Client({'d':[0,0,0,1],'t':''}),
            corrections=[{'question':'nod','answer':'no','text':'明示訂正'}])
        self.assertEqual(result['recognition']['accepted']['events']['nod']['value'],'yes')
        self.assertEqual(result['effective_events']['nod']['value'],'no')
        self.assertEqual(result['effective_events']['nod']['source'],'explicit_user')

    def test_visible_speech_needs_matching_pair_and_v4_group(self):
        compact = deepcopy(COMPACT)
        compact['features'] = [{'evidence_id': key} for key in ('E_asr0','E_mouth0','E_av_sync0','E_visible_speech0','E_head')]
        compact['experimental_candidates']['visible_person_speaks'] = ['E_asr0','E_mouth0']
        options = decision_options(compact)
        value = decode_decisions([1,0,0,0],options,compact)
        self.assertEqual(value['events']['visible_person_speaks']['evidence_ids'],['E_asr0','E_mouth0'])
        compact['experimental_candidates']['visible_person_speaks'] = ['E_asr0','E_mouth1']
        self.assertEqual(len(decision_options(compact)[0]),1)
        compact['recognition_contract'] = 'av-visible-v1'
        compact['experimental_candidates']['visible_person_speaks'] = ['E_asr0','E_mouth0']
        # A legacy pair cannot replace the required measured AV group.
        self.assertEqual(len(decision_options(compact)[0]),1)
        with self.assertRaisesRegex(ValueError,'compact_decision_index'):
            decode_decisions([1,0,0,0],decision_options(compact),compact)

    def test_unknown_transport_and_extra_wire_fields_rejected(self):
        with self.assertRaisesRegex(ValueError,'unsupported_cooperative_transport'):
            DriverCooperativePipeline(Client()).process(COMPACT,transport='invented')
        result = self.call(Client({'d':[0,0,0,0],'t':'','verified':True}))
        self.assertEqual(result['recognition']['status'],'invalid_output')

    def test_code_transport_retains_evidence_and_rejects_invalid_indices(self):
        options = decision_options(COMPACT)
        self.assertEqual(code_choices(options),['0000','0001'])
        for value in ('0002','0100','000',0,'000A'):
            with self.assertRaises(ValueError): decode_code(value,options,COMPACT)
        value = decode_code('0001',options,COMPACT)
        self.assertEqual(value['events']['nod']['evidence_ids'],['E_head'])
        client = Client({'d':'0001','t':''})
        result = DriverCooperativePipeline(client).process(COMPACT,transport='compact-cycle-v2')
        self.assertEqual(result['recognition']['status'],'accepted')
        self.assertEqual(result['transport_contract'],'compact-cycle-v2')

    def test_projection_preserves_supported_observations_and_source(self):
        compact = deepcopy(COMPACT)
        compact['features'] += [{'evidence_id':'E_ser','kind':'acoustic_emotion','value':'unverified'}]
        compact['limitations'] = ['speaker_unverified','uncalibrated_geometry']
        before = deepcopy(compact)
        projected = project_candidate_observations(compact,decision_options(compact))
        self.assertEqual(compact,before)
        self.assertEqual(projected['features'],[{'evidence_id':'E_head'}])
        self.assertEqual(projected['limitations'],compact['limitations'])
        self.assertEqual(len(projected['source_compact_sha256']),64)

    def test_string_wire_returns_real_text_and_same_validated_evidence(self):
        text = '休憩の際は、安全な場所に停車してから休みましょう。'
        client = Client('1|'+text)
        result = DriverCooperativePipeline(client).process(COMPACT,transport='compact-cycle-v3',
            preferences=[{'source':'explicit_user','preference':'rest_information'}])
        self.assertEqual(result['response']['status'],'accepted')
        self.assertEqual(result['response']['accepted']['text'],text)
        self.assertEqual(result['recognition']['accepted']['events']['nod']['evidence_ids'],['E_head'])
        self.assertEqual(client.calls[0][1],{'type':'string','enum':['0|'+text,'1|'+text]})
        self.assertEqual(json.loads(client.calls[0][0][1]['content'])['code_table'],['0000','0001'])
        payload = json.loads(client.calls[0][0][1]['content'])
        self.assertEqual(payload['decision_options'],[[[]],[[]],[[]],[[],['E_head']]])
        self.assertEqual(payload['observations']['features'],COMPACT['features'])
        self.assertEqual(len(result['recognition']['raw']['input_projection']['source_compact_sha256']),64)

    def test_string_wire_rejects_bad_index_text_and_late_output_as_a_whole(self):
        for value in ('01|','-1|','2|','true|','0','0|あなたは眠いです。',{'d':'0000','t':''}):
            with self.subTest(value=value):
                result = DriverCooperativePipeline(Client(value)).process(COMPACT,transport='compact-cycle-v3')
                self.assertEqual(result['recognition']['status'],'invalid_output')
                self.assertIsNone(result['recognition']['accepted'])
                self.assertFalse(result['response']['generated_text_adopted'])
        result = DriverCooperativePipeline(Client('0|',sleep=.03)).process(
            COMPACT,transport='compact-cycle-v3',timeout_s=.01)
        self.assertEqual(result['recognition']['status'],'inference_failure')

    def test_string_wire_preserves_explicit_silence_and_corrections(self):
        result = DriverCooperativePipeline(Client('1|')).process(COMPACT,transport='compact-cycle-v3',
            preferences=[{'source':'explicit_user','preference':'no_conversation'}],
            corrections=[{'question':'nod','answer':'no','text':'明示訂正'}])
        self.assertEqual(result['response']['status'],'policy_output')
        self.assertEqual(result['response']['accepted']['text'],'')
        self.assertEqual(result['recognition']['accepted']['events']['nod']['value'],'yes')
        self.assertEqual(result['effective_events']['nod']['source'],'explicit_user')

    def test_startup_warmup_has_no_sensor_input_and_is_not_a_display_result(self):
        client = Client('0|')
        result = warmup_compact_cycle(client)
        self.assertEqual(result['input']['features'],[])
        self.assertTrue(all(not ids for ids in result['input']['experimental_candidates'].values()))
        self.assertEqual(result['purpose'],'synthetic_startup_only_never_display')
        self.assertNotIn('response',result)
        self.assertEqual(result['cycle']['recognition']['status'],'accepted')

    def test_string_wire_keeps_av_group_and_rejects_shifted_speech_candidate(self):
        from tests.test_av_visible_speech import compact
        from modules.av_visible_speech import apply_contract
        source = apply_contract(compact())
        client = Client('1|')
        result = DriverCooperativePipeline(client).process(source,transport='compact-cycle-v3')
        event = result['recognition']['accepted']['events']['visible_person_speaks']
        self.assertEqual(event['value'],'yes')
        features = json.loads(client.calls[0][0][1]['content'])['observations']['features']
        self.assertTrue(set(event['evidence_ids']) <= {f['evidence_id'] for f in features})
        self.assertTrue(any(f['evidence_id']=='E_speech_mouth0' for f in features))
        shifted = apply_contract(compact(lag=480))
        rejected = DriverCooperativePipeline(Client('1|')).process(shifted,transport='compact-cycle-v3')
        self.assertEqual(rejected['recognition']['status'],'invalid_output')
        self.assertIsNone(rejected['recognition']['accepted'])

    def test_summary_keeps_untrusted_speech_and_limits_and_validates_full_source(self):
        from tests.test_av_visible_speech import compact
        from modules.av_visible_speech import apply_contract
        source=apply_contract(compact());before=deepcopy(source)
        source['limitations']=['speaker_unverified','absolute_av_sync_unmeasured']
        client=Client('1|')
        result=DriverCooperativePipeline(client).process(source,transport='compact-cycle-v4')
        self.assertEqual(result['recognition']['status'],'accepted')
        payload=json.loads(client.calls[0][0][1]['content'])
        summary=payload['observations']['evidence_summary']
        self.assertEqual(summary['E_asr0']['text'],'untrusted sensor words')
        self.assertTrue(summary['E_speech_mouth0']['attribution_unresolved'])
        self.assertEqual(payload['observations']['limitations'],source['limitations'])
        self.assertEqual(summary['E_av_sync0'],next(f for f in source['features'] if f['evidence_id']=='E_av_sync0'))
        self.assertFalse(result['recognition']['raw']['input_projection']['full_candidate_features_sent'])
        source.pop('limitations');self.assertEqual(source,before)
        rejected=DriverCooperativePipeline(Client('2|')).process(source,transport='compact-cycle-v4')
        self.assertIsNone(rejected['recognition']['accepted'])


if __name__ == '__main__': unittest.main()
