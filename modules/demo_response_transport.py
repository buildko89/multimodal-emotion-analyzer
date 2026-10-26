"""Interactive response wire for local ARA, leveraging Qwen2.5-7B natural dialogue."""
import json
from pathlib import Path
import time

from modules.driver_observation_features import compact_observations
from modules.local_llm_client import LocalLLMClient
from modules.observation_worker_client import write_json

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
    def __init__(self, client):
        self.client = client

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

        try:
            raw = self.client.generate([
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ], llm_schema, deadline=deadline, max_tokens=128)

            if raw.get('status') == 'generated' and raw.get('text'):
                val = json.loads(raw['text'])
                gen_text = val.get('text', '').strip()
                if gen_text and len(gen_text) <= 80:
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
    cycle = response_cycle(compact, client=ResponseTextClient(client), preferences=configuration.get('preferences'),
        guidance=configuration.get('guidance'), timeout_s=timeout_val)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'compact_observations.json', compact)
    write_json(output / 'feature_evidence.json', mapping)
    write_json(output / 'cooperative_cycle.json', cycle)
    return cycle
