"""Preparation must fail closed and must never create a live result or preference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

from modules.live_preparation import prepare_runtime, preparation_status
from modules.live_pipeline import run_live


class PreparationTests(unittest.TestCase):
    def setup_case(self, root):
        output = root/'run'; output.mkdir()
        paths = {}
        for name in ('video','audio'):
            path = root/name; path.write_bytes(b'saved-input')
            paths[name] = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        config = {'live':{'startup_preparation':{'saved_inputs':paths}},
                  'output':{},'display':{},'logging':{},'observation_integration':{},
                  'cooperative_ai':{'base_url':'http://127.0.0.1:18081','model':'test',
                                    'preferences':[{'source':'explicit_user','preference':'minimal_guidance'}]}}
        def run(**kwargs):
            (output/'preparation/observation_envelope_v1.json').write_text(json.dumps({
                'backend_health':{'visual':'ok','sensevoice':'ok','emotion2vec':'ok'}}))
        app = types.SimpleNamespace(run=Mock(side_effect=run),observation_session=Mock())
        runtime = types.SimpleNamespace(output=output,new_app=Mock(return_value=app),detector=Mock())
        return config, runtime, app

    def test_saved_preparation_separate_budgets_and_no_live_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, runtime, app = self.setup_case(Path(tmp)); original=deepcopy(config)
            cycle = {'cycle':{'recognition':{'status':'accepted'},'response':{'status':'accepted'},'wall_seconds':4}}
            with patch('modules.compact_driver_cycle.warmup_compact_cycle', return_value=cycle) as warm:
                result = prepare_runtime(runtime, config, Path(tmp))
            self.assertTrue(result['ready'])
            self.assertEqual([c.kwargs['timeout_s'] for c in warm.call_args_list],[60,10])
            self.assertEqual(config, original)
            self.assertEqual(runtime.new_app.call_args.kwargs, {'preparation':True})
            runtime.detector.reset_window_state.assert_called_once()
            app.observation_session.close.assert_called_once()
            self.assertFalse((runtime.output/'windows').exists())
            self.assertFalse((runtime.output/'result.json').exists())
            self.assertFalse((runtime.output/'capture_command.json').exists())
            self.assertFalse(result['used_live_input'])

    def test_bounded_check_failure_cannot_publish_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,runtime,_ = self.setup_case(Path(tmp))
            ok = {'cycle':{'recognition':{'status':'accepted'},'response':{'status':'accepted'},'wall_seconds':4}}
            fail = {'cycle':{'recognition':{'status':'inference_failure'},'response':{'status':'fallback'},'wall_seconds':10}}
            with patch('modules.compact_driver_cycle.warmup_compact_cycle',side_effect=[ok,fail]):
                with self.assertRaisesRegex(RuntimeError,'ara_contract'): prepare_runtime(runtime,config,Path(tmp))
            state=json.loads((runtime.output/'preparation_state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['state'],'failed'); self.assertFalse(state['ready'])
            self.assertEqual(len(state['phases']),3)

    def test_changed_input_rejected_before_model_or_ara(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,runtime,_ = self.setup_case(Path(tmp))
            config['live']['startup_preparation']['saved_inputs']['audio']['sha256']='bad'
            with patch('modules.compact_driver_cycle.warmup_compact_cycle') as warm:
                with self.assertRaisesRegex(ValueError,'hash_mismatch'): prepare_runtime(runtime,config,Path(tmp))
            runtime.new_app.assert_not_called(); warm.assert_not_called()
            self.assertFalse(json.loads((runtime.output/'preparation_state.json').read_text())['ready'])

    def test_backend_failure_never_calls_ara_and_resets_person(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,runtime,app = self.setup_case(Path(tmp))
            app.run.side_effect=lambda **kw: (runtime.output/'preparation/observation_envelope_v1.json').write_text(
                json.dumps({'backend_health':{'sensevoice':'failed'}}))
            with patch('modules.compact_driver_cycle.warmup_compact_cycle') as warm:
                with self.assertRaisesRegex(RuntimeError,'backend_unhealthy'): prepare_runtime(runtime,config,Path(tmp))
            warm.assert_not_called(); runtime.detector.reset_window_state.assert_called_once()

    def test_preparation_requires_reusable_manual_capture(self):
        for live in ({}, {'persistent_windows':True}, {'persistent_windows':True,'record_on_request':True,'ara_startup_warmup':True}):
            with self.subTest(live=live), self.assertRaisesRegex(ValueError,'preparation_requires'):
                run_live({'live':{**live,'startup_preparation':{'saved_inputs':{}}}})

    def test_display_has_distinct_failure_and_ready_instructions(self):
        self.assertIn('準備完了',preparation_status('ready'))
        self.assertIn('準備回答後',preparation_status('ready'))
        self.assertIn('開始しません',preparation_status('failed'))
        self.assertIn('保存素材',preparation_status('perception'))

    def test_camera_cannot_arm_recording_while_preparation_is_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'run'
            process=Mock(); process.pid=123; process.returncode=None
            process.poll.return_value=None
            process.wait.side_effect=lambda **kw: setattr(process,'returncode',0)
            class Display:
                frames=0
                def __init__(self,*a):pass
                def draw(self,snapshot,save=False):
                    if save:(output/'display_latest.png').write_bytes(b'png')
                    return {'readback_checked':save}
                def close(self):pass
            class Microphone:
                def __init__(self,*a):pass
                def run(self,stop,*a):stop.wait(2)
            def frames(stop,*a):
                (output/'capture_command.json').write_text(json.dumps({'request_id':'unapproved','not_before_ns':time.monotonic_ns()}))
                while not stop.is_set():
                    now=time.monotonic_ns()
                    yield {'native_ns':now,'available_ns':now,'bgr':None}
                    stop.wait(.01)
            config={'live':{'output':str(output),'font_path':'unused','duration_seconds':2,
                'warmup_seconds':0,'record_on_request':True,'persistent_windows':True,
                'startup_preparation':{'saved_inputs':{}},'preparation_deadline_seconds':.2,
                'worker_deadline_seconds':1},'cooperative_ai':{}}
            with patch('modules.live_pipeline.WORKSPACE',root), patch(
                'modules.framebuffer_display.FramebufferDisplay',Display), patch(
                'modules.live_pipeline.NativeMicrophone',Microphone), patch(
                'modules.live_pipeline.camera_frames',frames), patch(
                'modules.live_pipeline.subprocess.Popen',return_value=process), patch(
                'modules.live_capture.ManualCaptureGate.arm') as arm:
                report=run_live(config)
            arm.assert_not_called()
            self.assertTrue(all(report['threads_stopped'].values()))
            self.assertEqual(report['preparation_state'],'failed')
            self.assertIn('preparation_deadline',report['preparation_error'])
            final_state=json.loads((output/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(final_state['preparation_state'],'failed')
            events=[json.loads(line) for line in (output/'live_events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(e['event']=='window_failure' for e in events))
            self.assertFalse(any(e['event'] in ('window_closed','manual_recording_started','processing_started') for e in events))


if __name__=='__main__':unittest.main()
