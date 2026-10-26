"""CLI JSON is not proof of an actual generated video or human reference."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[2]/'devai/devai_ara/scripts/ara04_gemini_cli.py'
spec=importlib.util.spec_from_file_location('ara04_gemini_cli',PATH)
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)


def reviewed():
    return {'status':'reviewed','cases':[{'id':i,'prompt_en':'fixed camera fictional adult',
        'prompt_ja':'架空の成人','negative_prompt':'captions, cuts','review_notes':'one action'} for i in bridge.IDS]}


class GeminiBridgeTests(unittest.TestCase):
    def test_success_without_structured_review_is_rejected(self):
        for value in ({'status':'SUCCESS','response':'generated four videos'},
                      {'status':'WAITING','structured_output':reviewed()},
                      {'status':'ERROR','response':'permission denied'}):
            with self.assertRaises(ValueError): bridge.validate_envelope(value)

    def test_duplicate_cases_commands_and_missing_prompts_are_rejected(self):
        changes=[]
        value=reviewed();value['cases'][1]['id']='A01';changes.append(value)
        value=reviewed();value['cases'][0]['execute']='Remove-Item';changes.append(value)
        value=reviewed();value['cases'][0]['prompt_en']='';changes.append(value)
        value=reviewed();value['cases'].pop();changes.append(value)
        for value in changes:
            with self.assertRaises(ValueError): bridge.validate_review(value)

    def test_unknown_outside_workspace_path_rejected(self):
        with self.assertRaises(ValueError): bridge.workspace(Path('C:/Windows'))

    def test_only_prompt_data_returned_from_successful_review(self):
        value=reviewed();before=deepcopy(value)
        self.assertEqual(bridge.validate_envelope({'status':'SUCCESS','structured_output':value}),before)
        self.assertEqual(value,before)


if __name__=='__main__':unittest.main()
