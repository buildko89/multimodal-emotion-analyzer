"""File decode must preserve timestamps, frame identity, native defaults and cleanup."""
from pathlib import Path
import unittest
from unittest.mock import patch

from modules.file_video_decode import BACKEND,BitexactCapture,open_video


class DecodeTests(unittest.TestCase):
    def test_native_default_and_camera_selection(self):
        with patch('cv2.VideoCapture',return_value='native') as capture:
            self.assertEqual(open_video('saved.mp4'),'native')
            self.assertEqual(open_video(0,BACKEND),'native')
            self.assertEqual(capture.call_count,2)

    def test_unknown_backend_rejected(self):
        with self.assertRaisesRegex(ValueError,'unsupported'):open_video('saved.mp4','typo')

    def test_missing_file_releases_iterator(self):
        with self.assertRaises(Exception):BitexactCapture('this-file-does-not-exist.mp4')

    def test_bitexact_rgb_matches_probe_and_preserves_vfr_pts(self):
        import av
        import numpy as np
        from av.video.reformatter import Interpolation
        source=Path(__file__).resolve().parents[2]/'devai/devai_ara/runs/ara05_pc_live_20261018_v1/comparison_inputs/pc_live_8s/video.mp4'
        cap=BitexactCapture(source)
        try:
            import cv2
            with av.open(str(source)) as container:
                for index,frame in enumerate(container.decode(video=0)):
                    ok,bgr=cap.read();self.assertTrue(ok)
                    expected=frame.to_ndarray(format='rgb24',interpolation=Interpolation.BILINEAR|Interpolation.ACCURATE_RND|Interpolation.BITEXACT)
                    self.assertTrue(np.array_equal(bgr[:,:,::-1],expected))
                    self.assertAlmostEqual(cap.get(cv2.CAP_PROP_POS_MSEC)/1000,float(frame.pts*frame.time_base),places=9)
                self.assertEqual(index+1,122)
                self.assertEqual(cap.read(),(False,None))
        finally:cap.release()
        self.assertFalse(cap.isOpened())

    def test_helper_matches_direct_and_exits(self):
        import numpy as np
        source=Path(__file__).resolve().parents[2]/'devai/devai_ara/runs/ara05_pc_live_20261018_v1/comparison_inputs/board_saved_8s/video.mp4'
        import sys
        installed=sys.executable
        direct=BitexactCapture(source)
        with patch('modules.file_video_decode.sys.executable','worker-without-pyav.exe'):
            helper=BitexactCapture(source,installed)
        try:
            for _ in range(80):
                a,x=direct.read();b,y=helper.read();self.assertTrue(a and b)
                self.assertTrue(np.array_equal(x,y))
            self.assertEqual(helper.read(),(False,None))
        finally:
            direct.release();helper.release()
        self.assertIsNotNone(helper.proc.poll())

    def test_early_close_stops_owned_helper(self):
        import sys
        installed=sys.executable
        source=Path(__file__).resolve().parents[2]/'devai/devai_ara/runs/ara05_pc_live_20261018_v1/comparison_inputs/board_saved_8s/video.mp4'
        with patch('modules.file_video_decode.sys.executable','worker-without-pyav.exe'):
            helper=BitexactCapture(source,installed)
        helper.read();helper.release()
        self.assertIsNotNone(helper.proc.poll())


if __name__=='__main__':unittest.main()
