"""Catch FunASR constructor/generate overriding the requested CPU budget."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave
import numpy  # Keep the real module loaded outside patch.dict's restoration.

spec=importlib.util.spec_from_file_location('thread_budget_worker',Path(__file__).resolve().parents[1]/'scripts/observation_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


class AudioThreadBudgetTests(unittest.TestCase):
    def exercise(self,cached):
        current=[4]
        torch=SimpleNamespace(set_num_threads=lambda n:current.__setitem__(0,n),get_num_threads=lambda:current[0])
        class Adapter:
            def __init__(self):self.backend=None;self.load_report={};self.loads=0
            def load(self):
                if self.backend is None:
                    self.loads+=1;current[0]=4
                    self.backend=SimpleNamespace(kwargs={'ncpu':4},_base_kwargs_map={'kwargs':{'ncpu':4}})
            def analyze(self,*args,**kwargs):
                self.load()
                self.backend.kwargs=dict(self.backend._base_kwargs_map['kwargs'])
                current[0]=self.backend.kwargs['ncpu']
                return {'status':'ok','observations':[]}
        adapter=Adapter()
        if cached:adapter.load()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pcm=b'\0\0'*1600
            with wave.open(str(root/'audio.wav'),'wb') as stream:
                stream.setparams((1,2,16000,0,'NONE','not compressed'));stream.writeframes(pcm)
            request={'input_path':str(root/'audio.wav'),'sample_interval':[0,1600],
                'pcm_sha256':hashlib.sha256(pcm).hexdigest(),'audio_offset_seconds':0,
                'interval_seconds':[0,.1],'backend_id':'sensevoice','clip_id':'test','torch_num_threads':2}
            adapters={'sensevoice':adapter} if cached else {}
            with patch.dict('sys.modules',{'torch':torch,'sensevoice_adapter':SimpleNamespace(SenseVoiceAdapter=lambda:adapter)}):
                worker.audio(request,root,adapters)
            artifact=json.loads((root/'audio_observation.json').read_text(encoding='utf-8'))
            self.assertEqual(artifact['torch_num_threads'],2)
            self.assertEqual(adapter.loads,1)
            self.assertEqual(artifact['model_reused'],cached)
            self.assertIs(adapters['sensevoice'],adapter)

    def test_cold_load_and_generate_respect_request(self):self.exercise(False)
    def test_cached_generate_respects_request(self):self.exercise(True)
