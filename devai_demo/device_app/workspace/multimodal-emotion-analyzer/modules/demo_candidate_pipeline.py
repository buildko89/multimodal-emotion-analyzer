"""Opt-in demo candidate path. Keeps frozen evaluators and thresholds unchanged."""
from copy import deepcopy
import time

from modules.av_visible_speech import CONTRACT, groups
from modules.driver_observation_features import compact_observations, rule_recognition
from modules.driver_recognition import validate_recognition
from modules.driver_response_policy import select_policy, produce_response, validate_response

VERSION = 'demo-observation-response-v1'
CANDIDATES = (('C1_visual', ('visual',)), ('C1_audio', ('audio',)),
              ('C1_asr', ('asr',)), ('C2_legacy', ('visual', 'audio')), ('C2', ('visual', 'audio')))


def supported_rule_recognition(compact):
    narrowed = deepcopy(compact)
    support = narrowed['experimental_candidates']['visible_person_speaks']
    if compact.get('recognition_contract') != CONTRACT:
        pairs = [[i, 'E_mouth'+i[5:]] for i in support
                 if i.startswith('E_asr') and 'E_mouth'+i[5:] in support]
        narrowed['experimental_candidates']['visible_person_speaks'] = sorted(pairs)[0] if pairs else []
    recognition = rule_recognition(narrowed)
    validate_recognition(recognition, compact)
    return recognition


def collect_candidates(envelope_path, thresholds, *, builder=None, recognizer=None):
    builder = builder or compact_observations
    recognizer = recognizer or supported_rule_recognition
    result = {}
    for name, modalities in CANDIDATES:
        contract = CONTRACT if name == 'C2' else 'legacy-v3'
        try:
            compact, evidence = builder(envelope_path, thresholds, modalities, recognition_contract=contract)
            recognition = recognizer(compact)
            validate_recognition(recognition, compact)
            result[name] = dict(status='accepted_candidate', recognition=recognition,
                                compact=compact, evidence=evidence)
        except (ValueError, KeyError, TypeError, OSError) as error:
            result[name] = dict(status='inference_failure', error_type=type(error).__name__, reason=str(error))
    return dict(schema='demo-candidate-collection-v1', version=VERSION, candidates=result,
                human_reference_used=False, independent_quality=None)


def response_cycle(compact, *, client=None, preferences=None, guidance=None, timeout_s=30):
    """Rules retain observations; the optional LLM generates companion dialogue text."""
    if not 0 < timeout_s <= 60:
        raise ValueError('demo_cycle_deadline_must_be_at_most_sixty_seconds')
    started = time.monotonic()
    accepted = supported_rule_recognition(compact)
    recognition = dict(status='accepted', accepted=accepted, source='checked_observation_rules',
                       quality_accepted=False, validation=validate_recognition(accepted, compact))
    policy = select_policy(recognition, preferences, guidance)
    fallback = dict(schema='ara-response-v1', policy=policy['policy'], text=policy['text'])
    if client is None:
        response = dict(status='policy_output', accepted=fallback, validation=validate_response(fallback, policy),
                        llm_called=False, generated_text_adopted=False)
    else:
        try:
            if hasattr(client, 'generate_companion_response'):
                response = client.generate_companion_response(policy, compact, deadline=started+timeout_s)
            else:
                response = produce_response(client, policy, deadline=started+timeout_s)
        except (OSError, ValueError, TimeoutError) as error:
            response = dict(status='fallback', accepted=fallback, validation=validate_response(fallback, policy),
                            llm_called=True, generated_text_adopted=False, rejection=str(error))
    return dict(schema='ara-cooperative-cycle-v1', transport_contract=VERSION,
        recognition=recognition, policy=policy, response=response,
        effective_events={q: dict(value=e['value'], source='machine_candidate') for q,e in accepted['events'].items()},
        wall_seconds=time.monotonic()-started, timeout_seconds=timeout_s,
        recognition_corrections=[], self_report=None, human_reference_used=False,
        capabilities=dict(observation_candidates=True, llm_can_override_events=False,
                          driver_state_quality=False, negative_events_verified=False, speaker_attribution_verified=False))
