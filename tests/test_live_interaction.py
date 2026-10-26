from copy import deepcopy
import json
import io
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from modules.live_interaction import LiveInteraction, atomic_json, poll_commands, text_hash
from modules.live_pipeline import adopt_response, window_configuration, process_window
from modules.live_window_runtime import serve_windows


BASE = [{'source':'explicit_user','preference':'minimal_guidance','automatic_from_asr':False}]


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.state = LiveInteraction(BASE,'必要な案内だけ表示します。',session_id='session')

    def command(self, **changes):
        return {'request_id':'id'+str(self.state.revision),'source':'explicit_user',
                'input_channel':'conversation_user_reply','scope':'actual','automatic_from_asr':False,
                'expected_revision':self.state.revision,'kind':'preference_set',
                'preference':'no_conversation',**changes}

    def test_set_correct_clear_preserves_original_and_does_not_resurrect(self):
        initial=deepcopy(BASE)
        for command in (self.command(),):self.state.apply(command,10)
        self.assertEqual(self.state.response['text'],'')
        self.state.apply(self.command(preference='minimal_guidance'),20)
        self.assertEqual(self.state.response['text'],'必要な案内だけ表示します。')
        self.state.apply(self.command(kind='preference_clear',preference=None),30)
        self.assertEqual(self.state.response['text'],'')
        self.assertEqual(len(self.state.history),4)
        self.assertEqual(BASE,initial)

    def test_untrusted_channels_scope_stale_and_duplicates_do_not_mutate(self):
        command=self.command()
        for changes in ({'source':'asr'},{'automatic_from_asr':True},{'scope':'fixture'},
                        {'input_channel':'asr'},{'expected_revision':False},{'expected_revision':3}):
            before=self.state.snapshot()
            with self.assertRaises(ValueError):self.state.apply({**command,**changes},10)
            self.assertEqual(self.state.snapshot(),before)
        self.state.apply(command,10)
        with self.assertRaises(ValueError):self.state.apply(command,20)

    def test_asr_correction_requires_matching_window_and_preserves_raw_and_preference(self):
        self.state.observe_asr('会話をやめて','window1',1)
        before=deepcopy(self.state.history)
        command=self.command(kind='asr_correction',window_id='window1',raw_text_sha256=text_hash('会話をやめて'),text='画面を見せて')
        self.state.apply(command,10)
        self.assertEqual(self.state.asr['asr_text'],'会話をやめて')
        self.assertEqual(self.state.asr['asr_correction']['text'],'画面を見せて')
        self.assertEqual(self.state.history,before)
        self.state.observe_asr('','window2',20)
        self.assertNotIn('asr_correction',self.state.snapshot())
        with self.assertRaises(ValueError):self.state.apply({**command,'request_id':'new','expected_revision':1},30)

    def test_inflight_old_revision_cannot_overwrite_new_policy(self):
        lock=threading.Lock();ready=threading.Event();release=threading.Event();shared={}
        old={'preference_revision':self.state.preference_revision,'text':'古い案内','source_end_ns':1}
        outcomes=[]
        def worker():
            ready.set();release.wait(2)
            with lock:outcomes.append(adopt_response(shared,self.state,old))
        worker_thread=threading.Thread(target=worker);worker_thread.start();ready.wait(2)
        with lock:
            self.state.apply(self.command(),10)
            shared.update(self.state.snapshot())
        release.set();worker_thread.join(2)
        self.assertFalse(worker_thread.is_alive())
        self.assertEqual(outcomes,[False]);self.assertEqual(shared['response']['text'],'')

    def test_new_revision_result_adopts_and_future_commands_are_not_preapplied(self):
        result={'preference_revision':self.state.preference_revision,'text':'検証済み短文'}
        shared={};self.assertTrue(adopt_response(shared,self.state,result))
        result['text']='mutated';self.assertEqual(shared['response']['text'],'検証済み短文')
        self.state.observe_asr('希望を解除して','w',1)
        self.assertEqual(self.state.preference_revision,'session:0')

    def test_immutable_window_configuration_does_not_follow_later_changes(self):
        config={'cooperative_ai':{'preferences':deepcopy(BASE)},'live':{}}
        with tempfile.TemporaryDirectory() as folder:
            path=window_configuration(config,self.state,folder)
            self.state.apply(self.command(),10)
            value=json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(value['live']['preference_revision'],'session:0')
            self.assertEqual(value['cooperative_ai']['preferences'],BASE)
        self.assertNotIn('preference_revision',config['live'])

    def test_file_inbox_rejects_asr_and_replay_without_losing_valid_input(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder);(output/'interaction_commands').mkdir();events=[]
            atomic_json(output/'interaction_commands/01.json',self.command(source='asr'))
            atomic_json(output/'interaction_commands/02.json',self.command())
            poll_commands(self.state,output,10,events.append)
            poll_commands(self.state,output,20,events.append)
            self.assertEqual(self.state.revision,1);self.assertEqual(len(events),2)
            self.assertFalse(json.loads((output/'interaction_acks/01.json').read_text(encoding='utf-8'))['accepted'])
            self.assertTrue(json.loads((output/'interaction_acks/02.json').read_text(encoding='utf-8'))['accepted'])

    def test_fixture_commands_cannot_enter_actual_session(self):
        with self.assertRaises(ValueError):
            self.state.apply(self.command(scope='fixture',input_channel='fixed_test_fixture'),10)
        fixture=LiveInteraction(BASE,'案内',scope='fixture')
        fixture.apply(self.command(scope='fixture',input_channel='fixed_test_fixture'),10)
        self.assertEqual(self.state.history,BASE)
        self.assertIn('試験用',fixture.snapshot()['preference_label'])

    def test_processing_propagates_frozen_revision_to_production_and_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);manifest=folder/'capture_manifest.json'
            atomic_json(manifest,{'closed_available_ns':1,'interval_ns':[0,1],
                'audio_offset_seconds':0,'window_id':'w','video':'v','audio_file':'a'})
            config={'live':{'preference_revision':'frozen:2','diagnostic_replay':False},
                'output':{},'display':{},'logging':{},'synchronization':{},'observation_integration':{},
                'cooperative_ai':{'preferences':BASE}}
            atomic_json(folder/'processing_config.json',config)
            class Session:
                def close(self):pass
            class App:
                observation_session=Session()
                def run(self,**kwargs):atomic_json(folder/'app/observation_envelope_v1.json',{'input':{}})
            runtime=types.SimpleNamespace(new_app=lambda p:App())
            def process(envelope,output,configuration):
                self.assertEqual(configuration['preferences'],BASE)
                return {'response':{'accepted':{'text':'案内'},'status':'accepted'},'recognition':{'status':'accepted'}}
            with patch.dict('sys.modules',{'multimodal_app':types.SimpleNamespace(MultiModalEmotionApp=object)}), patch(
                    'modules.driver_response_policy.run_saved_observations',side_effect=process):
                process_window(manifest,folder/'processing_config.json',runtime=runtime)
            for name in ('production_display.json','result.json'):
                self.assertEqual(json.loads((folder/name).read_text(encoding='utf-8'))['preference_revision'],'frozen:2')

    def check_persistent_request(self, *, wrong_config=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'output';window=output/'windows/w1';window.mkdir(parents=True)
            config=root/'live.json';atomic_json(config,{'live':{'output':str(output)}})
            frozen=window/'processing_config.json'
            if wrong_config:frozen=output/'windows/w2/processing_config.json'
            request={'manifest':str(window/'capture_manifest.json'),'config':str(frozen)}
            noop=lambda *args:None
            tensorflow=types.SimpleNamespace(config=types.SimpleNamespace(threading=types.SimpleNamespace(
                set_intra_op_parallelism_threads=noop,set_inter_op_parallelism_threads=noop)))
            runtime=types.SimpleNamespace(close=noop)
            with patch.dict('sys.modules',{'tensorflow':tensorflow}),patch('signal.signal'),patch(
                    'modules.live_pipeline.WORKSPACE',root),patch('modules.live_pipeline.process_window') as process,patch(
                    'modules.live_window_runtime.LiveWindowRuntime',return_value=runtime),patch(
                    'sys.stdin',io.StringIO(json.dumps(request)+'\n')):
                if wrong_config:
                    with self.assertRaisesRegex(ValueError,'window_config_outside'):serve_windows(config)
                    process.assert_not_called()
                else:
                    serve_windows(config)
                    process.assert_called_once_with(window/'capture_manifest.json',frozen,runtime=runtime)

    def test_persistent_runtime_passes_frozen_config_to_processing(self):
        self.check_persistent_request()

    def test_persistent_runtime_rejects_other_window_config(self):
        self.check_persistent_request(wrong_config=True)


if __name__=='__main__':unittest.main()
