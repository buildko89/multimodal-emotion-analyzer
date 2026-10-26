"""Present existing compact visual observations with explicit emotion and state recognition for exhibition HUD."""

EMOTION_JA = {
    'happy': '笑顔・喜び',
    'neutral': '真顔・平静',
    'sad': '疲労・気落ち',
    'angry': '緊張・ストレス',
    'surprise': '驚き・注視',
    'fear': '警戒・不安',
    'disgust': '不快・違和感',
    'yawn': 'あくび・眠気兆候'
}

EMOTION_BADGES = {
    'happy': '[笑顔]',
    'neutral': '[真顔]',
    'sad': '[疲労]',
    'angry': '[緊張]',
    'surprise': '[注視]',
    'fear': '[警戒]',
    'disgust': '[不快]',
    'yawn': '[眠気]'
}


def visual_observation_result(compact, window_id, source_end_ns, available_ns):
    """Extract observations and aggregate dominant emotion and driver states."""
    if compact.get('schema') != 'ara-compact-observations-v1':
        raise ValueError('unsupported_visual_display_schema')
    features = {f['kind']: f['value'] for f in compact.get('features', [])}
    face = features.get('face_summary', {})
    visibility = features.get('visibility_summary', {})
    hand = features.get('projected_hand_mouth', {})
    head = features.get('head_pitch_geometry', {})

    def count(value):
        return value if type(value) is int and value >= 0 else None

    candidates = head.get('excursion_return_candidates')
    valid = count(head.get('valid_samples'))
    hands = count(hand.get('hand_detected_frames')) or 0

    # 感情認識集計
    raw_counts = face.get('raw_class_counts', {})
    total_face = sum(raw_counts.values()) if raw_counts else 0
    dominant_emotion = 'neutral'
    dominant_ratio = 0.0
    if total_face > 0:
        dominant_emotion = max(raw_counts, key=raw_counts.get)
        dominant_ratio = raw_counts[dominant_emotion] / total_face

    # 手検出と表情を組み合わせた眠気・あくび判定
    is_yawn = (hands >= 5 and dominant_emotion in ('sad', 'neutral', 'angry'))
    if is_yawn:
        display_emotion = 'yawn'
        display_emotion_ja = 'あくび・眠気兆候'
    else:
        display_emotion = dominant_emotion
        display_emotion_ja = EMOTION_JA.get(dominant_emotion, '真顔・平静')

    return dict(state='available', window_id=window_id, source_end_ns=source_end_ns,
                available_ns=available_ns, frames=count(face.get('samples')),
                single_face_frames=count(face.get('single_face_samples')),
                geometry_frames=count(visibility.get('single_face_frames')),
                missing_geometry_frames=count(visibility.get('missing_geometry_frames')),
                hand_detected_frames=hands,
                head_valid_samples=valid,
                head_candidate_count=len(candidates) if valid and isinstance(candidates, list) else None,
                head_observation_diagnostic=head.get('observation_diagnostic'),
                raw_counts=raw_counts,
                dominant_emotion=dominant_emotion,
                dominant_ratio=round(dominant_ratio * 100, 1),
                display_emotion=display_emotion,
                display_emotion_ja=display_emotion_ja,
                is_yawn=is_yawn,
                source_kind='completed_live_window', recognition_truth_verified=False)


def visual_display_lines(snapshot, now_ns):
    """Render provenance, pending, emotion classification and driver state lines."""
    heading = '【感情・画像認識（観測）】'
    if snapshot.get('input_faults'):
        return [heading, '入力停止中：画像結果は非表示']
    result = snapshot.get('visual_result')
    if not result:
        return [heading, '撮影後に結果を表示します']
    end = result.get('source_end_ns')
    if type(end) is not int or type(now_ns) is not int or now_ns < end:
        return [heading, '対象時刻が不明：画像結果は非表示']
    age = (now_ns-end)/1e9
    target = f'対象：{str(result.get("window_id", "不明"))[:24]}／{age:.1f}秒前'
    if result.get('state') == 'processing':
        return [heading, target, '画像結果を処理中']
    if result.get('state') != 'available':
        return [heading, target, '画像結果は不明（観測を取得できません）']
    if age > snapshot.get('max_result_age_seconds', 300):
        return [heading, target, '前の画像結果は期限切れです']
    available = result.get('available_ns')
    if type(available) is not int or available > now_ns or available < end:
        return [heading, target, '結果の時刻が不明：画像結果は非表示']

    def number(key):
        value = result.get(key)
        return str(value) if type(value) is int and value >= 0 else '不明'

    # 表情認識ライン（絵文字を完全排除し、Meiryoで確実に表示できるバッジを採用）
    display_emotion = result.get('display_emotion', 'neutral')
    emotion_name = result.get('display_emotion_ja', '真顔・平静')
    ratio = result.get('dominant_ratio', 0.0)
    badge = EMOTION_BADGES.get(display_emotion, '[観測]')
    emotion_line = f'表情認識：{badge} {emotion_name} ({ratio:.0f}%)'

    # 内訳ライン
    raw_counts = result.get('raw_counts', {})
    sorted_counts = sorted(raw_counts.items(), key=lambda x: x[1], reverse=True)
    detail_parts = [f"{EMOTION_JA.get(k, k)[:2]}:{v}" for k, v in sorted_counts[:3] if v > 0]
    detail_line = f'内訳：{" / ".join(detail_parts)}' if detail_parts else f'顔検出：{number("single_face_frames")}枚'

    hands = result.get('hand_detected_frames')
    if type(hands) is not int or hands < 0:
        hand_line = '手の動作：観測不明'
    elif hands == 0:
        hand_line = '手の動作：未検出（通常姿勢）'
    else:
        hand_line = f'手の動作：検出 {hands}枚（口元動作あり）'

    candidates = result.get('head_candidate_count')
    diagnostic = result.get('head_observation_diagnostic') or {}
    missing = result.get('missing_geometry_frames')
    incomplete = (diagnostic.get('coverage_complete') is False
                  or type(missing) is not int or missing > 0)
    if type(candidates) is not int or candidates < 0:
        head_line = '頭部動作：観測不足'
    elif candidates:
        suffix = '欠測あり' if incomplete else '検知'
        head_line = f'頭部動作：うなずき/傾き候補あり {candidates}件（{suffix}）'
    elif incomplete:
        head_line = '頭部動作：観測中（途切れあり）'
    elif diagnostic.get('excursion_without_confirmed_return'):
        head_line = '頭部動作：傾き姿勢あり'
    else:
        head_line = '頭部動作：安定'

    return [heading, target, emotion_line, detail_line, hand_line, head_line]
