"""Causality, gaps, bounded work and display age without board hardware."""
import threading
import unittest
from modules.live_capture import LatestWindowQueue, TimedPCM, validate_closed_window, ManualCaptureGate
from modules.framebuffer_display import visible_response


class LiveTests(unittest.TestCase):
    def test_manual_recording_requires_scheduled_native_time_and_consumes_once(self):
        gate=ManualCaptureGate()
        self.assertFalse(gate.recording(100))
        command={'request_id':'user_speech_01','not_before_ns':200}
        gate.arm(command,100)
        self.assertFalse(gate.recording(199))
        self.assertTrue(gate.recording(200))
        gate.arm({'request_id':'other','not_before_ns':300},210)
        self.assertEqual(gate.finish(),'user_speech_01')
        gate.arm(command,220)
        self.assertFalse(gate.recording(300))
        gate.arm({'request_id':'second','not_before_ns':400},300)
        self.assertFalse(gate.recording(399))
        self.assertTrue(gate.recording(400))

    def test_manual_capture_rejects_invalid_and_expired_commands(self):
        for command in ({},{'request_id':'x','not_before_ns':True},
            {'request_id':'x','not_before_ns':-1},{'request_id':'x','not_before_ns':0},
            {'request_id':'x','not_before_ns':200_000_000_000}):
            with self.assertRaises(ValueError):ManualCaptureGate().arm(command,40_000_000_000)

    def test_latest_pending_replaces_old_and_keeps_gap(self):
        queue = LatestWindowQueue()
        first = {'window_id':'a','interval_ns':[0,10]}
        second = {'window_id':'b','interval_ns':[10,20]}
        queue.put(first)
        self.assertIs(queue.get(),first)
        queue.put(first)
        queue.put(second)
        self.assertEqual(queue.gaps[0]['window_id'],'a')
        self.assertIs(queue.get(),second)
        queue.close()
        self.assertIsNone(queue.get())
        with self.assertRaises(RuntimeError): queue.put(first)

    def test_stop_drops_pending_and_wakes_idle_consumer(self):
        queue=LatestWindowQueue();results=[]
        thread=threading.Thread(target=lambda:results.append(queue.get()))
        thread.start();queue.close();thread.join(1)
        self.assertEqual(results,[None])
        queue=LatestWindowQueue();queue.put({'window_id':'a','interval_ns':[1,2]})
        queue.close();self.assertEqual(queue.gaps[0]['reason'],'stopped_pending_window')

    def test_pcm_slice_uses_only_requested_samples_and_preserves_bytes(self):
        ring=TimedPCM(rate=10)
        ring.append(10**9,b'\x01\x00'*10,2*10**9)
        result=ring.slice(12*10**8,18*10**8,2*10**9)
        self.assertEqual(result['sample_count'],6)
        self.assertEqual(result['pcm'],b'\x01\x00'*6)
        self.assertEqual(result['first_sample_monotonic_ns'],12*10**8)
        with self.assertRaisesRegex(ValueError,'future_or_invalid'):
            ring.slice(10**9,3*10**9,2*10**9)
        with self.assertRaisesRegex(ValueError,'future_audio_availability'):
            ring.slice(10**9,15*10**8,18*10**8)

    def test_pcm_missing_and_discontinuous_input_is_not_silence(self):
        ring=TimedPCM(rate=10)
        ring.append(10**9,b'\x00\x00'*10,2*10**9)
        with self.assertRaisesRegex(ValueError,'not_available'):
            ring.slice(0,2*10**9,3*10**9)
        with self.assertRaisesRegex(ValueError,'audio_clock_gap'):
            ring.append(3*10**9,b'\x00\x00'*10,4*10**9)
        with self.assertRaisesRegex(ValueError,'future_audio_timestamp'):
            TimedPCM(rate=10).append(0,b'\x00\x00'*10,0)

    def test_future_or_reordered_frame_window_is_rejected(self):
        window={'interval_ns':[10,20],'closed_available_ns':30,
            'frames':[{'native_ns':10,'available_ns':15}],
            'audio':{'first_sample_monotonic_ns':10,'sample_count':1,'rate':10**9}}
        validate_closed_window(window,40)
        with self.assertRaisesRegex(ValueError,'future_or_invalid'):
            validate_closed_window(window,25)
        window['frames'][0]['native_ns']=21
        with self.assertRaisesRegex(ValueError,'invalid_native_frame'):
            validate_closed_window(window,40)

    def test_old_result_expires_and_future_result_never_displays(self):
        result={'source_end_ns':10**9,'text':'休憩案内'}
        self.assertEqual(visible_response(result,2*10**9,5),('休憩案内',1))
        self.assertIn('期限切れ',visible_response(result,10*10**9,5)[0])
        with self.assertRaisesRegex(ValueError,'future_display'):
            visible_response(result,0,5)


if __name__=='__main__': unittest.main()
