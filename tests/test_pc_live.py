"""PC clocks and common display must retain uncertainty, boundaries and failures."""
from pathlib import Path
import tempfile
import time
import unittest

from modules.pc_live import AudioClock,WindowsDisplay
from modules.live_capture import TimedPCM


class PCLiveTests(unittest.TestCase):
    def test_adc_sample_clock_preserves_contiguous_samples_and_original_clocks(self):
        clock=AudioClock()
        first=clock.map(1600,10,10.12,100_000_000_000)
        second=clock.map(1600,10.1,10.222,100_102_000_000)
        self.assertEqual(second['native_start_ns']-first['native_start_ns'],100_000_000)
        self.assertEqual(second['sample_interval'],[1600,3200])
        self.assertEqual(second['portaudio_current_seconds'],10.222)
        self.assertFalse(second['absolute_av_sync_measured'])
        ring=TimedPCM()
        for stamp in (first,second):ring.append(stamp['native_start_ns'],b'\0\0'*1600,stamp['available_ns'])
        clip=ring.slice(first['native_start_ns'],second['native_start_ns'],second['available_ns'])
        self.assertEqual(clip['sample_count'],1600)

    def test_clock_jump_nonfinite_and_future_audio_fail_closed(self):
        for adc,current in ((float('nan'),10),(0,10),(11,10),(10,float('inf'))):
            with self.subTest(adc=adc),self.assertRaises(ValueError):AudioClock().map(1600,adc,current,100_000_000_000)
        clock=AudioClock();clock.map(1600,10,10.12,100_000_000_000)
        with self.assertRaisesRegex(ValueError,'clock_gap'):clock.map(1600,10.5,10.62,100_500_000_000)
        clock=AudioClock();clock.map(1600,10,10.12,100_000_000_000)
        with self.assertRaisesRegex(ValueError,'future_samples'):clock.map(1600,10.1,10.22,100_000_000_000)

    def test_pc_render_uses_shared_asr_separation_and_expires_old_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            display=WindowsDisplay(tmp,'C:/Windows/Fonts/meiryo.ttc',headless=True)
            result=display.draw({'sensor_capture_enabled':True,'asr_text':'会話をやめて',
                'asr_window_id':'saved_fixture','asr_source_end_ns':time.monotonic_ns(),
                'preference_label':'本人の明示希望：必要最小限','max_result_age_seconds':1,
                'response':{'text':'old fixture','source_end_ns':time.monotonic_ns()-2_000_000_000,'status':'accepted'}},save=True)
            self.assertIn('期限切れ',result['text'])
            self.assertEqual(result['readback_scope'],'software_render_buffer')
            self.assertFalse(result['physical_screen_readback'])
            self.assertFalse(result['pc_window_shown'])
            self.assertTrue((Path(tmp)/'display_latest.png').read_bytes().startswith(b'\x89PNG'))
            display.close()

    def test_slow_adc_drift_is_reported_without_confusing_it_with_missing_blocks(self):
        clock=AudioClock()
        for index in range(1201):
            adc=10+index*.100025
            stamp=clock.map(1600,adc,adc+.12,100_000_000_000+round(index*.100025*1e9))
        self.assertAlmostEqual(stamp['sample_clock_drift_seconds'],.03,places=6)
        self.assertLess(abs(stamp['block_clock_deviation_seconds']),.001)
        self.assertIsNone(stamp['source_timestamp_ns'])
        self.assertEqual(stamp['timeline_safety_lag_ns'],50_000_000)


if __name__=='__main__':unittest.main()
