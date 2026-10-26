"""Independent evaluation gates and abstention-aware scoring (no inference inputs).

This module is an evaluator. Human references must never be imported by a
predictor. A method template is not a frozen evaluation dataset.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

QUESTIONS = ('visible_person_speaks', 'offscreen_voice', 'hand_over_mouth', 'nod')
HASH = re.compile(r'^[a-f0-9]{64}$')
VALID_PREDICTIONS = {'yes', 'no', 'unknown', 'invalid_output', 'inference_failure', 'missing_prediction'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=pairs,
                      parse_constant=nonfinite)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


def grouped_records(records):
    """Union every shared identity, session, source, byte hash and audio lineage.

    Subject IDs include offscreen speakers. IDs are global pseudonyms, never
    reset per session. Unknown identity is a blocking condition, not a group.
    """
    parent, seen, identities = {}, {}, {}
    def find(ident):
        while parent[ident] != ident:
            parent[ident] = parent[parent[ident]]
            ident = parent[ident]
        return ident
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    for record in records:
        ident = record['clip_id']
        if not isinstance(ident, str) or not ident or ident in parent:
            raise ValueError('Empty or duplicate clip identity')
        parent[ident] = ident
        participants = record.get('participant_ids')
        if not isinstance(participants, list) or not participants or any(not isinstance(p, str) or not p for p in participants):
            raise ValueError('Known global participant IDs are required')
        if record.get('primary_participant_id') not in participants:
            raise ValueError('Visible target participant must be identified separately')
        for name in ('session_id', 'source_group_id'):
            if not isinstance(record.get(name), str) or not record[name]:
                raise ValueError('Missing lineage: ' + name)
        if not HASH.fullmatch(record.get('media_sha256', '')):
            raise ValueError('Missing media SHA256')
        tokens = [('person', p) for p in participants]
        tokens += [('session', record['session_id']), ('source', record['source_group_id']),
                   ('media', record['media_sha256'])]
        for name in ('audio_pcm_sha256', 'parent_source_group_id'):
            if record.get(name):
                namespace = 'pcm' if name == 'audio_pcm_sha256' else 'source'
                if namespace == 'pcm' and not HASH.fullmatch(record[name]):
                    raise ValueError('Invalid PCM SHA256')
                tokens.append((namespace, record[name]))
        for token in tokens:
            if token in seen:
                union(ident, seen[token])
            else:
                seen[token] = ident
        identities[ident] = tokens
    groups = defaultdict(list)
    for ident in parent:
        groups[find(ident)].append(ident)
    result = {}
    for members in groups.values():
        members = sorted(members)
        group = 'G_' + canonical_sha(members)[:16]
        result[group] = members
    return dict(sorted(result.items()))


def validate_dataset(dataset, root=None):
    if dataset.get('schema') != 'ara04-dataset-v1' or not isinstance(dataset.get('records'), list):
        raise ValueError('Invalid dataset schema')
    records = dataset['records']
    groups = grouped_records(records)
    by_id = {r['clip_id']: r for r in records}
    for record in records:
        conditions = record.get('conditions_observed', [])
        if not isinstance(conditions, list) or any(not isinstance(c, str) or not c for c in conditions):
            raise ValueError('Observed conditions must be actual string tags')
    for members in groups.values():
        roles = {by_id[m].get('split') for m in members}
        if len(roles) != 1 or not roles.issubset({'calibration', 'test'}):
            raise ValueError('Related people/session/source/hash cross split boundaries')
    if root is not None:
        root = Path(root).resolve()
        for record in records:
            path = (root / record['relative_path']).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha(path) != record['media_sha256']:
                raise ValueError('Media path/hash mismatch: ' + record['clip_id'])
    return groups


def resolve_references(dataset, receipts):
    """Agreeing independent, blind, whole-clip ratings; conflicts stay unknown.

    No automatic adjudication, no scenario labels and no AI-filled references.
    """
    by_id = {r['clip_id']: r for r in dataset['records']}
    collected = defaultdict(dict)
    for receipt in receipts:
        if receipt.get('schema') != 'ara04-human-review-v1':
            raise ValueError('Invalid review schema')
        reviewer = receipt.get('reviewer_id')
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError('Reviewer identity required')
        for row in receipt.get('clips', []):
            ident = row.get('clip_id')
            if ident not in by_id or row.get('media_sha256') != by_id[ident]['media_sha256']:
                raise ValueError('Review is not bound to dataset media')
            if reviewer in collected[ident]:
                raise ValueError('Duplicate reviewer/clip; use a separately versioned corrected dataset')
            if any(row.get(k) is not True for k in ('whole_clip_reviewed', 'model_outputs_not_viewed', 'recording_instruction_not_viewed')):
                raise ValueError('Unverified blind whole-clip review')
            answers = row.get('answers')
            if not isinstance(answers, dict) or set(answers) != set(QUESTIONS) or any(not isinstance(v, str) or v not in {'yes', 'no', 'unknown'} for v in answers.values()):
                raise ValueError('Invalid human event answers')
            collected[ident][reviewer] = answers
    result = {}
    for ident in by_id:
        reviews = collected[ident]
        events = {}
        reasons = {}
        for question in QUESTIONS:
            values = [r[question] for r in reviews.values()]
            if len(values) >= 2 and len(set(values)) == 1:
                events[question] = values[0]
                reasons[question] = 'independent_raters_agree'
            else:
                events[question] = 'unknown'
                reasons[question] = 'missing_second_reviewer' if len(values) < 2 else 'unresolved_disagreement'
        result[ident] = {'media_sha256': by_id[ident]['media_sha256'], 'events': events,
                         'reviewer_ids': sorted(reviews), 'resolution': reasons}
    return result


def validate_method(protocol):
    if protocol.get('schema') != 'ara04-protocol-template-v1':
        raise ValueError('Unknown evaluation method')
    for split in ('calibration', 'test'):
        for name in ('participants', 'per_question_per_class'):
            value = protocol['minimums'][split][name]
            if type(value) is not int or value < 1:
                raise ValueError('Positive data requirements are mandatory')
    for name in ('agreement', 'coverage', 'macro_f1', 'maximum_false_positive_rate', 'C3_minimum_paired_agreement_gain'):
        value = protocol['pilot_adoption_targets'][name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid pilot target')
        if name != 'maximum_false_positive_rate' and value == 0:
            raise ValueError('Positive quality/coverage targets are mandatory')


def readiness(dataset, receipts, protocol, root=None):
    validate_method(protocol)
    groups = validate_dataset(dataset, root)
    refs = resolve_references(dataset, receipts)
    reasons, coverage = [], {}
    records = dataset['records']
    if not records:
        reasons.append('no_recorded_media')
    for record in records:
        if record.get('source_kind') != 'real_recorded':
            reasons.append(record['clip_id'] + ':not_real_recorded')
        if record.get('rights_local_evaluation_confirmed') is not True:
            reasons.append(record['clip_id'] + ':rights_not_confirmed')
        if record.get('not_used_for_development') is not True:
            reasons.append(record['clip_id'] + ':development_use_or_unknown')
        if record.get('audio_present') is not True:
            reasons.append(record['clip_id'] + ':missing_real_audio')
        if record.get('split') == 'test' and record.get('test_content_not_used_for_tuning') is not True:
            reasons.append(record['clip_id'] + ':test_already_viewed_for_tuning_or_unknown')
    for split in ('calibration', 'test'):
        subset = [r for r in records if r['split'] == split]
        people = sorted({r['primary_participant_id'] for r in subset})
        counts = {q: dict(Counter(refs[r['clip_id']]['events'][q] for r in subset)) for q in QUESTIONS}
        coverage[split] = {'clips': len(subset), 'participants': people, 'reference_counts': counts}
        minimum = protocol['minimums'][split]
        if len(people) < minimum['participants']:
            reasons.append(split + ':insufficient_participants')
        for question in QUESTIONS:
            for label in ('yes', 'no'):
                if counts[question].get(label, 0) < minimum['per_question_per_class']:
                    reasons.append(split + ':' + question + ':insufficient_' + label)
    return {'schema': 'ara04-readiness-v1', 'ready': not reasons, 'reasons': sorted(set(reasons)),
            'recorded_clips': len(records), 'groups': groups, 'coverage': coverage,
            'state_quality_ready': False, 'state_reason': 'no_compatible_validated_ordinal_state_estimator',
            'independent_accuracy': None}


def freeze(dataset, receipts, protocol, artifacts, root, calibration_complete=False):
    """Actual groups + references + exact runtime/code/config hashes before test.

    Call only after calibration. A freeze with no data or placeholder artifacts
    is rejected, rather than reporting a successful empty evaluation.
    """
    if calibration_complete is not True:
        raise ValueError('Actual calibration completion must be explicitly declared')
    status = readiness(dataset, receipts, protocol, root)
    if not status['ready']:
        raise ValueError('Dataset not ready: ' + ', '.join(status['reasons']))
    required = {'predictor_code', 'scorer_code', 'inference_harness', 'inference_config', 'model', 'runtime'}
    if not required.issubset({a.get('kind') for a in artifacts}):
        raise ValueError('Exact predictor, scorer, harness, configuration, model and runtime artifacts required')
    for artifact in artifacts:
        path = (Path(root) / artifact['relative_path']).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or sha(path) != artifact.get('sha256'):
            raise ValueError('Freeze artifact hash mismatch')
    return {'schema': 'ara04-frozen-protocol-v1', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
            'dataset_sha256': canonical_sha(dataset), 'references_sha256': canonical_sha(receipts),
            'method': protocol, 'groups': status['groups'], 'artifacts': artifacts,
            'calibration_finished': True, 'test_predictions_pending': True}


def verify_frozen(frozen, dataset, root):
    if frozen.get('schema') != 'ara04-frozen-protocol-v1' or canonical_sha(dataset) != frozen.get('dataset_sha256'):
        raise ValueError('Frozen dataset mismatch')
    validate_dataset(dataset, root)
    for artifact in frozen['artifacts']:
        path = (Path(root) / artifact['relative_path']).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or sha(path) != artifact['sha256']:
            raise ValueError('Frozen predictor artifact changed')


def seal_predictions(path, frozen, dataset, root):
    """No reference file is accepted or opened in this operation."""
    verify_frozen(frozen, dataset, root)
    predictions = read_json(path)
    if predictions.get('schema') != 'ara04-predictions-v1' or predictions.get('frozen_protocol_sha256') != canonical_sha(frozen):
        raise ValueError('Predictions not bound to actual frozen protocol')
    if not isinstance(predictions.get('candidates'), dict) or not predictions['candidates']:
        raise ValueError('No prediction candidates')
    if predictions.get('human_reference_used') is not False:
        raise ValueError('Predictor must explicitly exclude human references')
    return {'schema': 'ara04-prediction-seal-v1', 'prediction_sha256': sha(path),
            'frozen_protocol_sha256': canonical_sha(frozen),
            'sealed_at_utc': datetime.now(timezone.utc).isoformat()}


def event_metrics(pairs):
    """Keep all attempted clips, including unknown/failure, in denominators."""
    known = [(gold, pred) for gold, pred in pairs if gold in {'yes', 'no'}]
    counts = Counter((gold, pred) for gold, pred in known)
    total, yes, no = len(known), sum(g == 'yes' for g, _ in known), sum(g == 'no' for g, _ in known)
    correct = counts['yes', 'yes'] + counts['no', 'no']
    answered = sum(pred in {'yes', 'no'} for _, pred in known)
    def ratio(a, b):
        return a / b if b else None
    f1s = []
    for label, other in (('yes', 'no'), ('no', 'yes')):
        support = sum(g == label for g, _ in known)
        tp, fp = counts[label, label], counts[other, label]
        fn = support - tp  # unknown, invalid, failure and missing are not correct
        f1s.append(ratio(2*tp, 2*tp + fp + fn) if support else None)
    recall_yes, recall_no = ratio(counts['yes', 'yes'], yes), ratio(counts['no', 'no'], no)
    return {'eligible_reference_items': total, 'reference_unknown_items': len(pairs)-total,
            'reference_yes': yes, 'reference_no': no, 'correct': correct,
            'full_denominator_agreement': ratio(correct, total), 'answer_coverage': ratio(answered, total),
            'selective_agreement': ratio(correct, answered),
            'positive_recall_full': recall_yes, 'negative_recall_full': recall_no,
            'balanced_agreement_full': (recall_yes+recall_no)/2 if yes and no else None,
            'macro_f1_full': sum(f1s)/2 if all(v is not None for v in f1s) else None,
            'false_positive_rate_all_negatives': ratio(counts['no', 'yes'], no),
            'coverage_by_reference_class': {label: ratio(sum(g == label and p in {'yes', 'no'} for g, p in known), n)
                for label, n in (('yes', yes), ('no', no))},
            'prediction_counts': dict(Counter(pred for _, pred in known)),
            'confusion': {gold: {pred: counts[gold, pred] for pred in sorted(VALID_PREDICTIONS)} for gold in ('yes', 'no')}}


def score_sealed(path, seal, frozen, dataset, receipts, root):
    verify_frozen(frozen, dataset, root)
    if (seal.get('schema') != 'ara04-prediction-seal-v1' or seal.get('prediction_sha256') != sha(path)
            or seal.get('frozen_protocol_sha256') != canonical_sha(frozen)):
        raise ValueError('Prediction seal mismatch')
    if canonical_sha(receipts) != frozen['references_sha256']:
        raise ValueError('Frozen references changed; retain old result and version corrections')
    refs = resolve_references(dataset, receipts)
    predictions = read_json(path)
    if predictions.get('frozen_protocol_sha256') != canonical_sha(frozen) or predictions.get('human_reference_used') is not False:
        raise ValueError('Prediction protocol mismatch')
    records = [r for r in dataset['records'] if r['split'] == 'test']
    by_id = {r['clip_id']: r for r in records}
    report = {}
    outcomes = {}
    for candidate, rows in predictions['candidates'].items():
        indexed = {}
        for row in rows:
            ident = row.get('clip_id')
            if ident not in by_id or ident in indexed or row.get('media_sha256') != by_id[ident]['media_sha256']:
                raise ValueError('Duplicate, non-test, or misbound prediction row')
            indexed[ident] = row
        pairs, per_person, per_condition, per_question = [], defaultdict(list), defaultdict(list), defaultdict(list)
        outcome = {}
        for record in records:
            ident = record['clip_id']
            row = indexed.get(ident)
            events = row.get('events') if row else None
            valid = isinstance(events, dict) and set(events) == set(QUESTIONS) and all(isinstance(v, str) and v in VALID_PREDICTIONS for v in events.values())
            for question in QUESTIONS:
                prediction = events[question] if valid else ('missing_prediction' if row is None else 'invalid_output')
                gold = refs[ident]['events'][question]
                pair = (gold, prediction)
                pairs.append(pair); per_question[question].append(pair)
                outcome[ident + ':' + question] = {'gold': gold, 'prediction': prediction, 'correct': gold in {'yes', 'no'} and gold == prediction}
                per_person[record['primary_participant_id']].append(pair)
                for condition in record.get('conditions_observed', []):
                    per_condition[condition].append(pair)
        summary = event_metrics(pairs)
        summary['per_question'] = {q: event_metrics(p) for q, p in per_question.items()}
        summary['per_person'] = {p: event_metrics(v) for p, v in per_person.items()}
        summary['per_observed_condition'] = {c: event_metrics(v) for c, v in per_condition.items()}
        limits = frozen['method']['pilot_adoption_targets']
        reasons = []
        for question, metric in summary['per_question'].items():
            for key, minimum in [('full_denominator_agreement', limits['agreement']), ('answer_coverage', limits['coverage']), ('macro_f1_full', limits['macro_f1'])]:
                if metric[key] is None or metric[key] < minimum:
                    reasons.append(question + ':' + key)
            for label, coverage in metric['coverage_by_reference_class'].items():
                if coverage is None or coverage < limits['coverage']:
                    reasons.append(question + ':coverage_' + label)
            fp = metric['false_positive_rate_all_negatives']
            if fp is None or fp > limits['maximum_false_positive_rate']:
                reasons.append(question + ':false_positive_rate')
        summary['pilot_event_targets_met'] = not reasons
        summary['unmet_targets'] = reasons
        report[candidate] = summary; outcomes[candidate] = outcome
    paired = {}
    if 'C2' in outcomes:
        for candidate, outcome in outcomes.items():
            if candidate == 'C2': continue
            eligible = [key for key, value in outcome.items() if value['gold'] in {'yes', 'no'}]
            gains = sum(outcome[k]['correct'] and not outcomes['C2'][k]['correct'] for k in eligible)
            losses = sum(outcomes['C2'][k]['correct'] and not outcome[k]['correct'] for k in eligible)
            paired[candidate] = {'same_items': len(eligible), 'gains': gains, 'losses': losses,
                'agreement_difference': (gains-losses)/len(eligible) if eligible else None,
                'quality_adoption': 'requires gain target plus event targets, evidence validity, latency, and deployment checks'}
    return {'schema': 'ara04-independent-report-v1', 'test_clips': len(records), 'candidates': report,
            'paired_against_C2': paired, 'state_quality': {'status': 'not_measured', 'ordinal_error': None},
            'scope': 'small pilot event evaluation; not population accuracy or deployment acceptance',
            'frozen_protocol_sha256': canonical_sha(frozen), 'prediction_sha256': seal['prediction_sha256']}
