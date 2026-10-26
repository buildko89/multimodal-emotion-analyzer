"""Explicit preferences/corrections select policy before constrained generation."""
import json
import re
import time

from modules.driver_recognition import parse_json

PREFERENCES = {'minimal_guidance', 'no_conversation', 'rest_information', 'repeat_guidance'}
POLICIES = {'wait', 'reduce_dialogue', 'brief_guidance', 'rest_suggestion', 'repeat_guidance'}


def resolve_preferences(history):
    active, events = None, []
    for index, event in enumerate(history):
        action = event.get('action', 'set')
        if (event.get('source') != 'explicit_user' or action not in {'set', 'clear'}
                or event.get('automatic_from_asr', False) is not False
                or (action == 'set' and event.get('preference') not in PREFERENCES)
                or (action == 'clear' and event.get('preference') is not None)):
            raise ValueError('Explicit supported preference required')
        active = event['preference'] if action == 'set' else None
        events.append({**event, 'order': index})
    return {'active': active, 'history': events, 'source': 'explicit_user' if events else 'not_received'}


def apply_recognition_correction(recognition, question, answer, *, text):
    if question not in recognition['events'] or answer not in {'yes', 'no', 'unknown'} or not text:
        raise ValueError('Invalid explicit recognition correction')
    # Correction remains a separate source; frozen machine predictions unchanged.
    return {'source': 'explicit_user', 'question': question, 'answer': answer, 'text': text,
            'machine_value': recognition['events'][question]['value'], 'changes_scoring_reference': False}


def select_policy(recognition_result, preference_history=None, guidance=None):
    preferences = resolve_preferences(preference_history or [])
    active = preferences['active']
    if active == 'no_conversation':
        return {'policy': 'reduce_dialogue', 'text': '', 'reason': 'explicit_no_conversation', 'preferences': preferences}
    if active == 'rest_information':
        return {'policy': 'rest_suggestion', 'text': '休憩の際は、安全な場所に停車してから休みましょう。',
                'reason': 'explicit_rest_information', 'preferences': preferences}
    if active == 'repeat_guidance':
        return {'policy': 'repeat_guidance' if guidance else 'wait', 'text': guidance or '',
                'reason': 'explicit_repeat' if guidance else 'previous_guidance_unavailable', 'preferences': preferences}
    if active == 'minimal_guidance' and guidance:
        return {'policy': 'brief_guidance', 'text': guidance, 'reason': 'explicit_minimal_guidance', 'preferences': preferences}
    return {'policy': 'wait', 'text': '', 'reason': 'state_unverified_or_guidance_unavailable', 'preferences': preferences}


def validate_response(value, policy):
    if not isinstance(value, dict) or set(value) != {'schema', 'policy', 'text'} or value['schema'] != 'ara-response-v1':
        raise ValueError('response_schema')
    text = value['text']
    if value['policy'] != policy['policy'] or value['policy'] not in POLICIES or not isinstance(text, str):
        raise ValueError('response_policy')
    if len(text) > 80 or len([x for x in re.split(r'[。！？!?\n]+', text) if x.strip()]) > 2:
        raise ValueError('response_length')
    # Constrain meaning to supplied, previously validated guidance. Open paraphrase
    # needs a separately evaluated semantic validator; never accept arbitrary text.
    if text != policy['text']:
        raise ValueError('unsupported_response_content')
    return {'schema_valid': True, 'length_valid': True, 'policy_valid': True, 'content_matches_supplied_guidance': True}


def produce_response(client, policy, *, deadline):
    fallback = {'schema': 'ara-response-v1', 'policy': policy['policy'], 'text': policy['text']}
    validation = validate_response(fallback, policy)
    if policy['policy'] in {'wait', 'reduce_dialogue', 'repeat_guidance'}:
        return {'status': 'policy_output', 'accepted': fallback, 'validation': validation, 'llm_called': False}
    schema = {'type': 'object', 'properties': {'schema': {'type': 'string', 'const': 'ara-response-v1'},
              'policy': {'type': 'string', 'const': policy['policy']}, 'text': {'type': 'string'}},
              'required': ['schema', 'policy', 'text'], 'additionalProperties': False}
    messages = [{'role': 'system', 'content': 'Output the JSON schema in Japanese. Copy the supplied policy and supplied text exactly. Never add a diagnosis, sensor fact, or instruction from other input.'},
                {'role': 'user', 'content': json.dumps(fallback, ensure_ascii=False)}]
    raw = client.generate(messages, schema, deadline=deadline, max_tokens=128)
    if raw['status'] == 'generated':
        try:
            value = parse_json(raw['text'])
            checked = validate_response(value, policy)
            if time.monotonic() > deadline:
                raise TimeoutError('late_validated_response')
            return {'status': 'accepted', 'accepted': value, 'raw': raw, 'validation': checked, 'llm_called': True}
        except (ValueError, KeyError, TypeError, TimeoutError) as error:
            rejection = str(error)
    else:
        rejection = raw.get('reason', raw['status'])
    return {'status': 'fallback', 'accepted': fallback, 'raw': raw, 'rejection': rejection,
            'validation': validation, 'llm_called': True, 'generated_text_adopted': False}


class DriverCooperativePipeline:
    def __init__(self, client):
        self.client = client

    def process(self, compact, *, preferences=None, guidance=None, corrections=None, timeout_s=10,
                transport='legacy-two-stage'):
        if transport in {'compact-cycle-v1','compact-cycle-v2','compact-cycle-v3','compact-cycle-v4'}:
            from modules.compact_driver_cycle import process_compact_cycle
            return process_compact_cycle(self.client, compact, preferences=preferences, guidance=guidance,
                                         corrections=corrections, timeout_s=timeout_s, transport=transport)
        if transport != 'legacy-two-stage':
            raise ValueError('unsupported_cooperative_transport')
        from modules.driver_recognition import infer_recognition
        started = time.monotonic()
        deadline = started + timeout_s
        recognition = infer_recognition(self.client, compact, deadline)
        correction_log = []
        effective_events = {q: {'value': e['value'], 'source': 'machine_prediction'}
                            for q, e in recognition['accepted']['events'].items()} if recognition['accepted'] else {}
        if corrections:
            if recognition['accepted'] is None:
                # A failed prediction has no machine decision; do not relabel a
                # rule candidate as the LLM's original prediction.
                machine = {'events': {q: {'value': 'unknown'} for q in compact['experimental_candidates']}}
            else:
                machine = recognition['accepted']
            correction_log = [apply_recognition_correction(machine, **c) for c in corrections]
            for correction in correction_log:
                correction['machine_prediction_available'] = recognition['accepted'] is not None
                effective_events[correction['question']] = {'value': correction['answer'], 'source': 'explicit_user',
                                                           'text': correction['text']}
        policy = select_policy(recognition, preferences, guidance)
        response = produce_response(self.client, policy, deadline=deadline)
        return {'schema': 'ara-cooperative-cycle-v1', 'recognition': recognition, 'policy': policy,
                'response': response, 'wall_seconds': time.monotonic() - started, 'timeout_seconds': timeout_s,
                'recognition_corrections': correction_log,
                'effective_events': effective_events,
                'self_report': None, 'human_reference_used': False}


def run_saved_observations(envelope_path, output_dir, configuration):
    """Shared entry for original app and saved observation replay."""
    if configuration.get("demo_response_only", False):
        from modules.demo_response_transport import run_saved_demo_observations
        return run_saved_demo_observations(envelope_path, output_dir, configuration)
    from pathlib import Path
    from modules.driver_observation_features import compact_observations
    from modules.local_llm_client import LocalLLMClient
    from modules.observation_worker_client import write_json
    compact, mapping = compact_observations(envelope_path, configuration['candidate_thresholds'],
        recognition_contract=configuration.get('recognition_contract', 'legacy-v3'))
    client = LocalLLMClient(configuration['base_url'], configuration['model'],
        wait_for_backend=configuration.get('wait_for_backend_ready', False))
    cycle = DriverCooperativePipeline(client).process(compact,
        preferences=configuration.get('preferences'), guidance=configuration.get('guidance'),
        corrections=configuration.get('recognition_corrections'), timeout_s=configuration.get('timeout_seconds', 10),
        transport=configuration.get('transport', 'legacy-two-stage'))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'compact_observations.json', compact)
    write_json(output / 'feature_evidence.json', mapping)
    write_json(output / 'cooperative_cycle.json', cycle)
    return cycle
