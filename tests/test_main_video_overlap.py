"""App-level deferred fusion parity and failure behavior without model weights."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from multimodal_app import MultiModalEmotionApp


class OverlapTests(unittest.TestCase):
    def run_app(self, directory, overlap, faces, *, asr_failure=False, video_failure=False):
        app = MultiModalEmotionApp.__new__(MultiModalEmotionApp)
        app.config = {'output': {'save_results': True, 'save_frame_observations': True},
                      'display': {'show_window': False},
                      'synchronization': {'audio_offset_seconds': -0.25},
                      'observation_integration': {'enabled': True, 'parallel_backends': True,
                          'asr_backend': 'sensevoice', 'overlap_main_video': overlap}}
        app.output_dir = Path(directory)
        app.speech_result = None
        app.observation_session = None
        app.frame_count = 0
        app.prev_time = 0
        app.logger = Mock()
        app.visualizer = SimpleNamespace(draw=lambda frame,*args: frame)
        app.multimodal_analyzer = Mock()
        app.multimodal_analyzer._emotion_only_state.return_value = {}
        app.multimodal_analyzer.analyze.return_value = {'same_analysis': True}
        app._log_emotion_change = Mock()
        app._print_final_statistics = Mock()
        calls = []
        def detect(frame):
            calls.append('video')
            if video_failure:
                raise RuntimeError('video_failed')
            return deepcopy(faces[len(calls)-1 if overlap else len(calls)-2])
        app.emotion_detector = SimpleNamespace(detect=detect)
        def speech():
            calls.append('asr')
            if asr_failure:
                return {'text': '', 'segments': [], 'status': 'inference_failure', 'reason':'timeout'}
            return {'text': 'machine', 'language':'ja', 'segments':[
                {'start':0, 'end':1.0,'text':'machine'}]}
        session = SimpleNamespace(common_audio=Path('common.wav'), sensevoice_transcript=speech,
                                  finish=Mock(return_value={'observations_by_kind':{},'backend_health':{}}))
        cap = Mock()
        cap.isOpened.return_value = True
        positions = iter([0,500])
        cap.get.side_effect = lambda prop: 2 if prop==cv2.CAP_PROP_FPS else next(positions)
        cap.read.side_effect = [(True,np.zeros((8,8,3),np.uint8))]*2+[(False,None)]
        with patch('modules.observation_integration.ObservationIntegrationSession',return_value=session), \
                patch('multimodal_app.cv2.VideoCapture',return_value=cap):
            if video_failure:
                with self.assertRaisesRegex(RuntimeError,'video_failed'):
                    app.process_with_audio_file('video.mp4','audio.wav')
            else:
                app.process_with_audio_file('video.mp4','audio.wav')
        cap.release.assert_called_once()
        return app, session, calls

    def test_deferred_fusion_matches_serial_for_aligned_absent_multiple_and_changed_tracks(self):
        face={'face_id':1,'emotion':'happy','confidence':.9,'scores':np.array([.1,.9])}
        for faces in ([[face],[face]], [[],[face]], [[face,dict(face,face_id=2)],[face]],
                      [[face],[dict(face,face_id=2)]]):
            with self.subTest(counts=[len(f) for f in faces]), tempfile.TemporaryDirectory() as directory:
                results=[]
                for overlap in (False,True):
                    folder=Path(directory)/str(overlap);folder.mkdir()
                    app,session,calls=self.run_app(folder,overlap,faces)
                    self.assertEqual(calls,['video','video','asr'] if overlap else ['asr','video','video'])
                    session.finish.assert_called_once()
                    summary=json.loads((folder/'video_summary.json').read_text())
                    origin=summary['timeline_origin_ns']
                    row=json.loads((folder/'fusion_windows.jsonl').read_text())
                    for key in ('window_start_ns','window_end_ns'): row[key]-=origin
                    for key in ('start_ns','end_ns'):row['speech_data'][key]-=origin
                    row['emotion_data'] and self.assertNotIn('driver_state',row['emotion_data'])
                    results.append(row)
                    timings=json.loads((folder/'pipeline_timing.json').read_text())
                    if overlap:
                        self.assertLessEqual(timings['video_loop_completed_ns'],timings['asr_wait_started_ns'])
                    self.assertEqual(summary['frames_processed'],2)
                self.assertEqual(*results)

    def test_asr_failure_produces_no_fabricated_speech_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            app,session,_=self.run_app(directory,True,[[],[]],asr_failure=True)
            self.assertEqual((Path(directory)/'fusion_windows.jsonl').read_text(),'')
            self.assertEqual(app.speech_result['status'],'inference_failure')
            self.assertEqual(session.finish.call_args.args[0]['reason'],'timeout')

    def test_video_failure_is_preserved_without_waiting_for_asr_or_finishing_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            app,session,calls=self.run_app(directory,True,[[],[]],video_failure=True)
            self.assertEqual(calls,['video'])
            session.finish.assert_not_called()
            summary=json.loads((Path(directory)/'video_summary.json').read_text())
            self.assertEqual(summary['completion'],'failed')
            self.assertFalse((Path(directory)/'speech_result.json').exists())
            self.assertTrue((Path(directory)/'pipeline_timing.json').exists())

    def test_overlap_rejects_missing_parallel_contract(self):
        app=MultiModalEmotionApp.__new__(MultiModalEmotionApp)
        app.config={'output':{'save_frame_observations':True},'observation_integration':{
            'enabled':True,'asr_backend':'sensevoice','overlap_main_video':True}}
        with self.assertRaisesRegex(ValueError,'requires parallel'):
            app.process_with_audio_file('video.mp4','audio.wav')


if __name__=='__main__':unittest.main()
