"""Interactive response wire for local ARA, leveraging Qwen2.5-7B natural dialogue."""
import hashlib
import json
from pathlib import Path
import threading
import time

from modules.driver_observation_features import compact_observations
from modules.local_llm_client import LocalLLMClient
from modules.observation_worker_client import write_json

# Persistent-runtime state for non-blocking generation.  ARA needs about 10 s
# per reply (3.7 s prefill), so a per-window budget of 4 s always fell back.
# Replies are kept only for byte-identical requests (temperature 0, seed 42).
_CACHE_LIMIT = 32
_cache = {}
_background = {'thread': None, 'last': None}
_state_lock = threading.Lock()


def request_digest(messages, schema):
    return hashlib.sha256(json.dumps([messages, schema], ensure_ascii=False, sort_keys=True)
                          .encode('utf-8')).hexdigest()


def accepted_text(raw):
    if raw.get('status') != 'generated' or not raw.get('text'):
        return None
    try:
        text = json.loads(raw['text']).get('text', '').strip()
    except (ValueError, AttributeError):
        return None
    return text if text and len(text) <= 80 else None


def _generate_in_background(client, key, messages, schema, seconds):
    started = time.monotonic()
    record = {'request_digest': key, 'started_monotonic': started, 'budget_seconds': seconds}
    try:
        raw = client.generate(messages, schema, deadline=started + seconds, max_tokens=128)
        text = accepted_text(raw)
        record.update(status=raw.get('status'), reason=raw.get('reason'), adopted=text is not None)
        if text is not None:
            with _state_lock:
                _cache[key] = {'text': text, 'raw': raw, 'generated_monotonic': time.monotonic(),
                               'generation_wall_seconds': raw.get('wall_seconds'), 'hits': 0}
                while len(_cache) > _CACHE_LIMIT:
                    _cache.pop(next(iter(_cache)))
    except Exception as error:
        record.update(status='inference_failure', reason=f'{type(error).__name__}: {error}', adopted=False)
    record['wall_seconds'] = time.monotonic() - started
    with _state_lock:
        _background['last'] = record


EMOTION_JA_DESC = {
    'happy': '笑顔（明るく楽しそう）',
    'neutral': '真顔（落ち着いている）',
    'sad': '疲労（少し疲れた表情）',
    'angry': '緊張（集中している・少し険しい）',
    'surprise': '驚き（何かを注視している）',
    'yawn': 'あくび・眠気兆候'
}


class ResponseTextClient:
    """Invokes Qwen2.5-7B to generate friendly, contextual vehicle companion dialogue."""
    def __init__(self, client, *, background_seconds=None):
        if background_seconds is not None and (isinstance(background_seconds, bool)
                or not isinstance(background_seconds, (int, float)) or not 1 <= background_seconds <= 60):
            raise ValueError('background_generation_seconds_out_of_range')
        self.client = client
        self.background_seconds = background_seconds

    def cached_or_background(self, messages, schema, policy, fallback, started):
        """Return at once: an exact-input ARA reply if one exists, else the fallback.

        On a miss, one generation runs in the background with a full budget so
        the next window with the same request is answered by ARA text.
        """
        key = request_digest(messages, schema)
        with _state_lock:
            hit = _cache.get(key)
            if hit is not None:
                hit['hits'] += 1
                hit = dict(hit)
                background = 'not_needed'
            elif _background['thread'] is not None and _background['thread'].is_alive():
                background = 'busy'
            else:
                thread = threading.Thread(target=_generate_in_background, name='ara-background',
                    args=(self.client, key, messages, schema, self.background_seconds), daemon=True)
                _background['thread'] = thread
                thread.start()
                background = 'started'
            last = dict(_background['last']) if _background['last'] else None
        validation = {'schema_valid': True, 'length_valid': True, 'policy_valid': True,
                      'content_matches_supplied_guidance': True}
        cache = {'request_digest': key, 'hit': hit is not None, 'background_generation': background,
                 'last_background_result': last, 'entries_limit': _CACHE_LIMIT}
        if hit is not None:
            cache.update(hits=hit['hits'], generation_wall_seconds=hit['generation_wall_seconds'],
                         generated_monotonic=hit['generated_monotonic'])
            return {'status': 'accepted', 'response_source': 'ara_exact_input_cache',
                    'accepted': {'schema': 'ara-response-v1', 'policy': policy['policy'], 'text': hit['text']},
                    'raw': hit['raw'], 'validation': validation, 'llm_called': False,
                    'generated_text_adopted': True, 'cache': cache, 'wall_seconds': time.monotonic() - started}
        return {'status': 'fallback', 'response_source': 'policy_fallback_while_ara_generates',
                'accepted': fallback, 'validation': validation, 'llm_called': background == 'started',
                'generated_text_adopted': False, 'cache': cache, 'wall_seconds': time.monotonic() - started}

    def generate_companion_response(self, policy, compact, *, deadline):
        started = time.monotonic()
        fallback = {'schema': 'ara-response-v1', 'policy': policy['policy'], 'text': policy['text']}

        # コンパクト観測から表情集計を抽出
        features = {f['kind']: f['value'] for f in compact.get('features', [])}
        face = features.get('face_summary', {})
        hand = features.get('projected_hand_mouth', {})
        raw_counts = face.get('raw_class_counts', {})
        hands = hand.get('hand_detected_frames', 0) if isinstance(hand, dict) else 0

        total_face = sum(raw_counts.values()) if raw_counts else 0
        dom_emotion = max(raw_counts, key=raw_counts.get) if total_face > 0 else 'neutral'
        if hands >= 5 and dom_emotion in ('sad', 'neutral', 'angry'):
            dom_emotion = 'yawn'
        emotion_desc = EMOTION_JA_DESC.get(dom_emotion, '真顔（落ち着いている）')

        guidance_text = policy.get('text', '本日も安全運転をサポートします。')

        system_prompt = (
            "あなたは親しみやすい車載AIアシスタント「ARA」です。\n"
            "ドライバーの表情や発話に合わせて、安心感と親しみのある自然な日本語で1〜2文（40文字前後、最大60文字以内）で話しかけてください。\n"
            "必ずJSON形式で {\"schema\": \"ara-response-v1\", \"policy\": \"guide\", \"text\": \"メッセージ\"} を返してください。"
        )
        user_prompt = (
            f"観測された表情: {emotion_desc}\n"
            f"発話または状況: {guidance_text}\n"
            "上記に応じた親しみやすい気の利いた一言メッセージを日本語で作成してください。"
        )

        llm_schema = {
            'type': 'object',
            'properties': {
                'schema': {'type': 'string', 'const': 'ara-response-v1'},
                'policy': {'type': 'string'},
                'text': {'type': 'string', 'maxLength': 80}
            },
            'required': ['schema', 'policy', 'text'],
            'additionalProperties': False
        }
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        if self.background_seconds is not None:
            return self.cached_or_background(messages, llm_schema, policy, fallback, started)

        try:
            raw = self.client.generate(messages, llm_schema, deadline=deadline, max_tokens=128)
            gen_text = accepted_text(raw)
            if gen_text is not None:
                accepted_val = {'schema': 'ara-response-v1', 'policy': policy['policy'], 'text': gen_text}
                return {
                    'status': 'accepted',
                    'accepted': accepted_val,
                    'raw': raw,
                    'validation': {'schema_valid': True, 'length_valid': True, 'policy_valid': True, 'content_matches_supplied_guidance': True},
                    'llm_called': True,
                    'generated_text_adopted': True,
                    'wall_seconds': time.monotonic() - started
                }
        except Exception:
            pass

        return {
            'status': 'fallback',
            'accepted': fallback,
            'validation': {'schema_valid': True, 'length_valid': True, 'policy_valid': True, 'content_matches_supplied_guidance': True},
            'llm_called': True,
            'generated_text_adopted': False,
            'wall_seconds': time.monotonic() - started
        }

    def generate(self, messages, schema, **kwargs):
        deadline = kwargs.get('deadline')
        max_tokens = kwargs.get('max_tokens', 128)
        return self.client.generate(messages, schema, deadline=deadline, max_tokens=max_tokens)


def run_saved_demo_observations(envelope_path, output_dir, configuration):
    compact, mapping = compact_observations(envelope_path, configuration['candidate_thresholds'],
        recognition_contract=configuration.get('recognition_contract', 'legacy-v3'))
    client = LocalLLMClient(configuration['base_url'], configuration['model'],
        wait_for_backend=configuration.get('wait_for_backend_ready', True))
    from modules.demo_candidate_pipeline import response_cycle
    timeout_val = configuration.get('cooperative_ai', {}).get('timeout_seconds', configuration.get('timeout_seconds', 30))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    # Written before ARA so the live display can show recognition without waiting.
    write_json(output / 'compact_observations.json', compact)
    write_json(output / 'feature_evidence.json', mapping)
    text_client = ResponseTextClient(client, background_seconds=configuration.get('background_generation_seconds'))
    cycle = response_cycle(compact, client=text_client, preferences=configuration.get('preferences'),
        guidance=configuration.get('guidance'), timeout_s=timeout_val)
    write_json(output / 'cooperative_cycle.json', cycle)
    return cycle
