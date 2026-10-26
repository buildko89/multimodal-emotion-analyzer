"""A production-only window must not execute a hidden diagnostic replay."""
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml
from modules.live_pipeline import process_window


class ProductionOnlyTests(unittest.TestCase):
    def test_diagnostic_is_optional_and_production_budget_stays_ten(self):
        for replay in (False,True):
            with self.subTest(replay=replay),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);folder=root/'window';folder.mkdir()
                manifest={'window_id':'measured','interval_ns':[1,2],
                    'closed_available_ns':time.monotonic_ns()-1,'audio_offset_seconds':0,
                    'video':'closed.mp4','audio_file':'closed.wav'}
                (folder/'capture_manifest.json').write_text(json.dumps(manifest))
                config={'live':{'diagnostic_replay':replay},'output':{},'display':{},'logging':{},
                    'synchronization':{},'observation_integration':{},'cooperative_ai':{'enabled':True}}
                path=root/'config.yaml';path.write_text(yaml.safe_dump(config))
                class App:
                    def run(self,**kwargs):
                        (folder/'app/observation_envelope_v1.json').write_text(json.dumps({'input':{}}))
                runtime=SimpleNamespace(new_app=lambda _:App())
                def generate(envelope,output,configuration):
                    return {'response':{'accepted':{'text':'必要な案内だけ表示します。'},'status':'accepted'},
                            'recognition':{'status':'accepted'}}
                with patch.dict(sys.modules,{'multimodal_app':SimpleNamespace(MultiModalEmotionApp=App)}),patch(
                    'modules.driver_response_policy.run_saved_observations',side_effect=generate) as calls:
                    process_window(folder/'capture_manifest.json',path,runtime=runtime)
                self.assertEqual(calls.call_count,2 if replay else 1)
                self.assertEqual(calls.call_args_list[0].args[2]['timeout_seconds'],10)
                result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
                self.assertEqual(result['diagnostic_executed'],replay)
                self.assertEqual(result['result_source'],'diagnostic_120s' if replay else 'production_10s')
