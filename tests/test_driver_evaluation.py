"""Functional fixtures only; these are not recorded driver evaluation data."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from modules import driver_evaluation as e


def record(ident, person, split='test', digest=None):
    return {'clip_id':ident,'primary_participant_id':person,'participant_ids':[person],
            'session_id':person+'_S1','source_group_id':ident,'media_sha256':digest or e.canonical_sha(ident),
            'audio_pcm_sha256':e.canonical_sha('audio_'+ident),'split':split,'source_kind':'real_recorded',
            'rights_local_evaluation_confirmed':True,'not_used_for_development':True,
            'test_content_not_used_for_tuning':True,'audio_present':True,'conditions_observed':['normal']}


def method():
    return {'schema':'ara04-protocol-template-v1',
            'minimums':{s:{'participants':1,'per_question_per_class':1} for s in ('calibration','test')},
            'pilot_adoption_targets':{'agreement':0.8,'coverage':0.8,'macro_f1':0.8,'maximum_false_positive_rate':0.1,'C3_minimum_paired_agreement_gain':0.05}}


def receipts(records):
    return [{'schema':'ara04-human-review-v1','reviewer_id':reviewer,'clips':[
        {'clip_id':r['clip_id'],'media_sha256':r['media_sha256'],'whole_clip_reviewed':True,
         'model_outputs_not_viewed':True,'recording_instruction_not_viewed':True,
         'answers':{q:'yes' if r['clip_id'].endswith('1') else 'no' for q in e.QUESTIONS}}
        for r in records]} for reviewer in ('R1','R2')]


class GroupTests(unittest.TestCase):
    def test_same_person_multiple_sessions_stay_one_group(self):
        a,b=record('A','P1'),record('B','P1');b['session_id']='P1_S2'
        self.assertEqual(len(e.grouped_records([a,b])),1)

    def test_transitive_person_and_source_links(self):
        a,b,c=record('A','P1'),record('B','P1'),record('C','P3')
        b['source_group_id']=c['source_group_id']
        self.assertEqual(len(e.grouped_records([a,b,c])),1)

    def test_offscreen_helper_cannot_cross_splits(self):
        a,b=record('A','P1','calibration'),record('B','P2')
        a['participant_ids'].append('HELPER');b['participant_ids'].append('HELPER')
        with self.assertRaisesRegex(ValueError,'cross split'):e.validate_dataset({'schema':'ara04-dataset-v1','records':[a,b]})

    def test_cropped_source_and_shared_pcm_join(self):
        a,b=record('A','P1'),record('B','P2');b['parent_source_group_id']=a['source_group_id']
        self.assertEqual(len(e.grouped_records([a,b])),1)
        b.pop('parent_source_group_id');b['audio_pcm_sha256']=a['audio_pcm_sha256']
        self.assertEqual(len(e.grouped_records([a,b])),1)

    def test_same_media_bytes_join(self):
        a,b=record('A','P1'),record('B','P2');b['media_sha256']=a['media_sha256']
        self.assertEqual(len(e.grouped_records([a,b])),1)

    def test_missing_identity_and_duplicate_clip_rejected(self):
        a=record('A','P1');a['participant_ids']=[]
        with self.assertRaises(ValueError):e.grouped_records([a])
        with self.assertRaises(ValueError):e.grouped_records([record('A','P1'),record('A','P2')])

    def test_missing_primary_not_filled_from_helper(self):
        a=record('A','P1');a.pop('primary_participant_id')
        with self.assertRaises(ValueError):e.grouped_records([a])


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.records=[record('A1','P1')];self.dataset={'schema':'ara04-dataset-v1','records':self.records}

    def test_single_reviewer_and_conflict_are_unknown(self):
        ratings=receipts(self.records)
        self.assertEqual(set(e.resolve_references(self.dataset,ratings[:1])['A1']['events'].values()),{'unknown'})
        ratings[1]['clips'][0]['answers']['nod']='no'
        result=e.resolve_references(self.dataset,ratings)['A1']
        self.assertEqual(result['events']['nod'],'unknown')
        self.assertEqual(result['resolution']['nod'],'unresolved_disagreement')

    def test_scenario_aware_review_is_rejected(self):
        ratings=receipts(self.records);ratings[0]['clips'][0]['recording_instruction_not_viewed']=False
        with self.assertRaisesRegex(ValueError,'blind'):e.resolve_references(self.dataset,ratings)

    def test_model_aware_review_is_rejected(self):
        ratings=receipts(self.records);ratings[0]['clips'][0]['model_outputs_not_viewed']=False
        with self.assertRaises(ValueError):e.resolve_references(self.dataset,ratings)

    def test_duplicate_reviewer_and_media_tamper_rejected(self):
        ratings=receipts(self.records)
        with self.assertRaisesRegex(ValueError,'Duplicate'):e.resolve_references(self.dataset,[ratings[0],ratings[0]])
        ratings[0]['clips'][0]['media_sha256']='0'*64
        with self.assertRaises(ValueError):e.resolve_references(self.dataset,ratings)

    def test_invalid_human_value_is_rejected(self):
        ratings=receipts(self.records);ratings[0]['clips'][0]['answers']['nod']={'value':'yes'}
        with self.assertRaises(ValueError):e.resolve_references(self.dataset,ratings)

    def test_empty_dataset_is_blocked_without_accuracy(self):
        result=e.readiness({'schema':'ara04-dataset-v1','records':[]},[],method())
        self.assertFalse(result['ready']);self.assertIsNone(result['independent_accuracy'])
        self.assertIn('no_recorded_media',result['reasons'])

    def test_synthetic_seen_unlicensed_missing_audio_are_not_independent(self):
        a=record('A1','P1');a.update(source_kind='synthetic_generated',not_used_for_development=False,
            rights_local_evaluation_confirmed=False,audio_present=False)
        result=e.readiness({'schema':'ara04-dataset-v1','records':[a]},receipts([a]),method())
        for suffix in ('not_real_recorded','development_use_or_unknown','rights_not_confirmed','missing_real_audio'):
            self.assertIn('A1:'+suffix,result['reasons'])

    def test_helpers_not_counted_as_independent_target_people(self):
        a=record('A1','P1');a['participant_ids'] += ['H1','H2','H3']
        result=e.readiness({'schema':'ara04-dataset-v1','records':[a]},receipts([a]),method())
        self.assertEqual(result['coverage']['test']['participants'],['P1'])

    def test_zero_minimum_and_nan_target_rejected(self):
        m=method();m['minimums']['test']['participants']=0
        with self.assertRaises(ValueError):e.validate_method(m)
        m=method();m['pilot_adoption_targets']['agreement']=float('nan')
        with self.assertRaises(ValueError):e.validate_method(m)


class ScoreTests(unittest.TestCase):
    def test_all_unknown_has_zero_full_score_no_false_true_negatives(self):
        result=e.event_metrics([('yes','unknown'),('no','unknown')])
        self.assertEqual(result['full_denominator_agreement'],0)
        self.assertEqual(result['answer_coverage'],0)
        self.assertEqual(result['negative_recall_full'],0)
        self.assertEqual(result['macro_f1_full'],0)
        self.assertIsNone(result['selective_agreement'])

    def test_empty_and_missing_reference_class_are_undefined(self):
        self.assertIsNone(e.event_metrics([])['full_denominator_agreement'])
        self.assertIsNone(e.event_metrics([('yes','yes')])['macro_f1_full'])

    def test_invalid_failure_missing_remain_denominator(self):
        result=e.event_metrics([('yes','yes'),('yes','inference_failure'),('no','invalid_output'),('no','missing_prediction'),('unknown','no')])
        self.assertEqual(result['eligible_reference_items'],4)
        self.assertEqual(result['reference_unknown_items'],1)
        self.assertEqual(result['full_denominator_agreement'],0.25)
        self.assertEqual(result['positive_recall_full'],0.5)

    def test_false_positive_counts_only_explicit_positive_on_negative(self):
        result=e.event_metrics([('no','yes'),('no','unknown'),('no','no')])
        self.assertAlmostEqual(result['false_positive_rate_all_negatives'],1/3)
        self.assertAlmostEqual(result['negative_recall_full'],1/3)

    def test_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.json'
            for text in ('{"a":1,"a":2}','{"a":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):e.read_json(path)

    def test_freeze_seal_score_bindings_missing_rows_and_artifact_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);records=[]
            for ident,person,split in [('A1','P1','calibration'),('A2','P1','calibration'),('B1','P2','test'),('B2','P2','test')]:
                path=root/(ident+'.bin');path.write_bytes(ident.encode())
                r=record(ident,person,split,e.sha(path));r['relative_path']=path.name;records.append(r)
            data={'schema':'ara04-dataset-v1','records':records};ratings=receipts(records)
            artifact=root/'artifact.bin';artifact.write_bytes(b'functional_fixture_only')
            artifacts=[{'kind':kind,'relative_path':artifact.name,'sha256':e.sha(artifact)} for kind in
                ('predictor_code','scorer_code','inference_harness','inference_config','model','runtime')]
            with self.assertRaisesRegex(ValueError,'calibration'):e.freeze(data,ratings,method(),artifacts,root)
            frozen=e.freeze(data,ratings,method(),artifacts,root,True)
            path=root/'predictions.json'
            predictions={'schema':'ara04-predictions-v1','frozen_protocol_sha256':e.canonical_sha(frozen),
                'human_reference_used':False,'candidates':{'C2':[{'clip_id':'B1','media_sha256':records[2]['media_sha256'],
                    'events':{q:'yes' for q in e.QUESTIONS}}],'C3':[{'clip_id':'B1','media_sha256':records[2]['media_sha256'],
                    'events':{q:'unknown' for q in e.QUESTIONS}}]}}
            e.write_new(path,predictions);seal=e.seal_predictions(path,frozen,data,root)
            report=e.score_sealed(path,seal,frozen,data,ratings,root)
            self.assertEqual(report['candidates']['C2']['full_denominator_agreement'],0.5)
            self.assertEqual(report['candidates']['C2']['prediction_counts']['missing_prediction'],4)
            self.assertFalse(report['candidates']['C2']['pilot_event_targets_met'])
            self.assertEqual(report['paired_against_C2']['C3']['losses'],4)
            changed=deepcopy(ratings);changed[0]['clips'][0]['answers']['nod']='no'
            with self.assertRaisesRegex(ValueError,'references changed'):e.score_sealed(path,seal,frozen,data,changed,root)
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'seal mismatch'):e.score_sealed(path,seal,frozen,data,ratings,root)
            artifact.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):e.verify_frozen(frozen,data,root)

    def test_empty_dataset_cannot_be_frozen(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'not ready'):
                e.freeze({'schema':'ara04-dataset-v1','records':[]},[],method(),[],Path(folder),True)


if __name__=='__main__':unittest.main()
