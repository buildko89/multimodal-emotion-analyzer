from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from modules.live_capture import camera_frames
from modules.live_health import capture_fault, stale_capture
from modules.live_pipeline import run_live


class FaultTests(unittest.TestCase):
    def test_capture_fault_keeps_reason_and_camera_or_mic_watchdog(self):
        state={'sensor_capture_enabled':True,'camera_native_ns':20_000_000_000,'audio_available_ns':20_000_000_000}
        self.assertIsNone(stale_capture(state,22_000_000_000,0))
        self.assertEqual(stale_capture(state,24_000_000_000,0),'camera')
        state['camera_native_ns']=24_000_000_000
        self.assertEqual(stale_capture(state,24_000_000_000,0),'microphone')
        state['sensor_capture_enabled']=False
        self.assertIsNone(stale_capture(state,100_000_000_000,0))
        capture_fault(state,'microphone','unplugged',25)
        self.assertEqual(state['input_faults'][0]['error'],'unplugged')
        self.assertIn('入力停止',state['status'])

    def test_camera_eof_is_a_failure_unless_stop_was_requested(self):
        container=types.SimpleNamespace(decode=lambda **k:iter(()))
        class Context:
            def __enter__(self):return container
            def __exit__(self,*args):pass
        with patch.dict('sys.modules',{'av':types.SimpleNamespace(open=lambda *a,**k:Context())}):
            stop=threading.Event()
            with self.assertRaisesRegex(RuntimeError,'stream_ended'):list(camera_frames(stop))
            stop.set();self.assertEqual(list(camera_frames(stop)),[])

    def test_camera_and_microphone_failure_stop_other_threads_and_close_display(self):
        for failed_role in ('camera','microphone'):
            with self.subTest(role=failed_role),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);snapshots=[];closed=[]
                class Display:
                    frames=0
                    def __init__(self,*args):pass
                    def draw(self,snapshot,save=False):snapshots.append(dict(snapshot));return {'readback_checked':save}
                    def close(self):closed.append(True)
                class Microphone:
                    def __init__(self,*args):pass
                    def run(self,stop,*args):
                        if failed_role=='microphone':raise RuntimeError('mic_disconnected')
                        stop.wait(2)
                def camera(stop,*args):
                    if failed_role=='camera':raise RuntimeError('camera_disconnected')
                    stop.wait(2)
                    return iter(())
                config={'live':{'output':str(root/'new_attempt'),'font_path':'unused','duration_seconds':2,
                        'record_on_request':True,'worker_deadline_seconds':1},'cooperative_ai':{}}
                with patch('modules.live_pipeline.WORKSPACE',root),patch(
                        'modules.framebuffer_display.FramebufferDisplay',Display),patch(
                        'modules.live_pipeline.NativeMicrophone',Microphone),patch(
                        'modules.live_pipeline.camera_frames',camera):
                    report=run_live(config)
                self.assertTrue(closed)
                self.assertTrue(all(report['threads_stopped'].values()))
                self.assertEqual(report['input_faults'][0]['role'],failed_role)
                self.assertIn('入力停止',snapshots[-1]['status'])

    def test_fault_view_hides_previous_accepted_guidance(self):
        import numpy as np
        from PIL import ImageFont
        from modules.framebuffer_display import FramebufferDisplay
        import time
        with tempfile.TemporaryDirectory() as tmp:
            display=FramebufferDisplay.__new__(FramebufferDisplay)
            color=lambda offset:types.SimpleNamespace(offset=offset,length=8)
            display.var=types.SimpleNamespace(xres=1920,yres=1080,xoffset=0,yoffset=0,
                red=color(16),green=color(8),blue=color(0),transp=types.SimpleNamespace(length=0))
            display.rows=np.zeros((1080,1920),dtype=np.uint32)
            display.fonts={size:ImageFont.load_default() for size in (24,32,42,54)}
            display.output=Path(tmp);display.frames=0
            result=display.draw({'input_faults':[{'role':'camera','error':'unplugged'}],
                'response':{'source_end_ns':time.monotonic_ns()-1_000_000_000,
                    'text':'古い固有の案内文テスト','status':'accepted','label':'ARA短文'}})
            self.assertNotIn('古い固有の案内文テスト',result['text'])
            self.assertIn('入力停止中',result['text'])
            self.assertIn('入力エラー',result['label'])


if __name__=='__main__':unittest.main()
