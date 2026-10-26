"""The optional ARM timestamp backend must preserve native PTS or reject input."""
from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from modules.media_timeline import probe_video_native

class NativeTimelineTests(unittest.TestCase):
    def container(self,pts=(0,1,2),duration=3):
        stream=SimpleNamespace(duration=duration,time_base=Fraction(1,24))
        container=SimpleNamespace(streams=SimpleNamespace(video=[stream]),
            decode=lambda _: iter(SimpleNamespace(pts=p,time_base=Fraction(1,24)) for p in pts))
        from unittest.mock import MagicMock
        context=MagicMock();context.__enter__.return_value=container
        return SimpleNamespace(open=lambda _:context,__version__='test')

    def test_preserves_rational_native_pts(self):
        with patch.dict('sys.modules',{'av':self.container()}):
            value=probe_video_native('clip','pyav')
        self.assertEqual([float(x['best_effort_timestamp_time']) for x in value['frames']],[0,1/24,2/24])
        self.assertEqual(float(value['streams'][0]['duration']),3/24)

    def test_rejects_missing_duration_without_fps_fallback(self):
        with patch.dict('sys.modules',{'av':self.container(duration=None)}),self.assertRaisesRegex(ValueError,'duration'):
            probe_video_native('clip','pyav')

    def test_rejects_missing_pts_without_index_fallback(self):
        with patch.dict('sys.modules',{'av':self.container(pts=(0,None))}),self.assertRaisesRegex(ValueError,'PTS'):
            probe_video_native('clip','pyav')

    def test_rejects_duplicate_or_reverse_timestamps(self):
        for pts in ((0,0),(1,0)):
            with self.subTest(pts=pts),patch.dict('sys.modules',{'av':self.container(pts=pts)}),self.assertRaises(ValueError):
                probe_video_native('clip','pyav')

    def test_rejects_outside_duration(self):
        with patch.dict('sys.modules',{'av':self.container(pts=(0,3))}),self.assertRaises(ValueError):
            probe_video_native('clip','pyav')

    def test_default_ffprobe_unchanged(self):
        with patch('modules.media_timeline.subprocess.check_output',return_value=b'{"frames":[],"streams":[]}') as call:
            self.assertEqual(probe_video_native('clip'),{'frames':[],'streams':[]})
        self.assertEqual(call.call_args.args[0][-1],'clip')
        self.assertEqual(call.call_args.args[0][0],'ffprobe')

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):probe_video_native('clip','fps')

if __name__=='__main__':unittest.main()
