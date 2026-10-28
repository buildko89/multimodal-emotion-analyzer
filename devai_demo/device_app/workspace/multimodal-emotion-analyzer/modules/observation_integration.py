"""Offline observation envelopes; no human labels or driver-state inference."""
from collections import Counter
import hashlib
import html
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path
import subprocess
import time
import wave

from modules.observation_worker_client import SubprocessObservationClient, digest, write_json

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
SCHEMA = 'ara-driver-observation-envelope-v1'
QUESTIONS = ('visible_person_speaks', 'offscreen_voice', 'hand_over_mouth', 'nod')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def person_windows(segments, visual, classification, *, offset_s, duration_s):
    """Time overlap is separate from speaker association and usable state input."""
    if not math.isfinite(offset_s) or not duration_s > 0:
        raise ValueError('Finite offset and duration required')
    windows = []
    for index, segment in enumerate(segments):
        start, end = segment['start'] + offset_s, segment['end'] + offset_s
        if not all(math.isfinite(v) for v in (start, end)) or end <= start:
            raise ValueError('Invalid speech interval')
        rows = [r for r in visual if start <= r['frame_time_s'] < end]
        faces = [r for r in classification if start <= r['source_pts_seconds'] < end]
        reasons = {'speaker_unverified', 'absolute_av_sync_unmeasured'}
        if start < 0 or end > duration_s:
            reasons.add('incomplete_media_window')
        if not rows:
            reasons.add('no_visual_observations')
        if any(r.get('face_count', 0) != 1 for r in rows) or not faces:
            reasons.add('face_count_or_classification_unavailable')
        if any(r.get('face_count', 0) > 1 for r in rows + faces):
            reasons.add('multiple_faces')
        tracks = {f['face_id'] for row in faces for f in row['faces']}
        if len(tracks) != 1:
            reasons.add('track_changed_or_unknown')
        if any(r.get('status') != 'ok' or r.get('hand_mouth', {}).get('status') == 'unknown' for r in rows):
            reasons.add('mouth_geometry_unknown')
        if any(r.get('hand_mouth', {}).get('status') == 'hand_overlap_risk' for r in rows):
            reasons.add('projected_hand_mouth_overlap')
        if any(r.get('eye_hand_risk', {}).get('hold_eye_scores', True) for r in rows):
            reasons.add('eye_visibility_unavailable')
        if any(b['frame_time_s'] - a['frame_time_s'] > 0.15 for a, b in zip(rows, rows[1:])):
            reasons.add('frame_gap')
        # Expose a conservative observation-level link, never a verified speaker.
        windows.append({'segment_index': index, 'interval_seconds': [start, end],
                        'time_alignment': 'overlapping' if rows else 'unavailable',
                        'speaker_link': {'status': 'unknown', 'track_id': None},
                        'screen_status': 'unknown', 'driver_state_usable': False,
                        'visual_samples': len(rows), 'uncertainty': sorted(reasons)})
    return windows


def validate_envelope(envelope, output_dir):
    """Structural, time and on-disk evidence validation. Not truth validation."""
    root = Path(output_dir).resolve()
    if envelope.get('schema') != SCHEMA or envelope.get('mode') != 'offline':
        raise ValueError('Invalid envelope schema/mode')
    required = {'run_id', 'clip_id', 'input', 'backend_manifest', 'backend_health', 'evidence_registry',
                'observations', 'person_windows', 'event_predictions', 'state_candidates', 'state_status',
                'self_report', 'preferences', 'response', 'human_reference_used'}
    if required - envelope.keys() or not all(isinstance(envelope[k], str) and envelope[k] for k in ('run_id', 'clip_id')):
        raise ValueError('Missing envelope fields or IDs')
    if not isinstance(envelope['backend_manifest'], dict) or envelope['human_reference_used'] is not False:
        raise ValueError('Invalid provenance/reference contract')
    json.dumps(envelope, allow_nan=False)
    start, end = envelope['input']['interval_seconds']
    if not 0 <= start < end:
        raise ValueError('Invalid media bounds')
    entries, file_cache = {}, {}
    for evidence in envelope['evidence_registry']:
        ident = evidence['evidence_id']
        if ident in entries:
            raise ValueError('Duplicate evidence ID')
        path = (root / evidence['relative_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Evidence path invalid')
        if path not in file_cache:
            file_cache[path] = {'sha': digest(path)}
        if file_cache[path]['sha'] != evidence['file_sha256']:
            raise ValueError('Evidence hash mismatch')
        left, right = evidence['media_interval_seconds']
        if not start <= left < right <= end + 1e-6:
            raise ValueError('Evidence interval outside media')
        if 'line_index' in evidence:
            lines = file_cache[path].setdefault('lines', None)
            if lines is None:
                lines = file_cache[path]['lines'] = read_lines(path)
            number = evidence['line_index']
            if isinstance(number, bool) or not isinstance(number, int) or not 0 <= number < len(lines):
                raise ValueError('Evidence line index invalid')
            row = lines[number]
            stamp = row.get('frame_time_s', row.get('source_pts_seconds'))
            if stamp is None or abs(stamp - left) > 1e-6:
                raise ValueError('Evidence frame time mismatch')
        else:
            value = file_cache[path].setdefault('json', None)
            if value is None:
                value = file_cache[path]['json'] = read_json(path)
            for key in evidence.get('json_path', []):
                value = value[key]
            if not isinstance(value, (dict, list)):
                raise ValueError('Evidence locator does not identify an observation')
        entries[ident] = evidence
    for observation in envelope['observations']:
        if not all(k in observation for k in ('kind', 'backend_id', 'status', 'value', 'quality', 'evidence_ids')):
            raise ValueError('Missing observation fields')
        if observation['status'] not in {'observed', 'unavailable', 'not_implemented', 'inference_failure'}:
            raise ValueError('Observation status invalid')
        if observation['quality']['status'] not in {'reported', 'not_measured', 'unavailable'}:
            raise ValueError('Quality status invalid')
        if any(key not in entries for key in observation['evidence_ids']):
            raise ValueError('Missing observation evidence')
        if observation['status'] == 'observed' and not observation['evidence_ids']:
            raise ValueError('Observed value requires evidence')
    if set(envelope['event_predictions']) != set(QUESTIONS):
        raise ValueError('Missing event slots')
    for prediction in envelope['event_predictions'].values():
        status = prediction['status']
        if status not in {'predicted', 'not_implemented', 'invalid_output', 'inference_failure'}:
            raise ValueError('Invalid event status')
        if status == 'predicted' and prediction['value'] not in {'yes', 'no', 'unknown'}:
            raise ValueError('Invalid event answer')
        if status != 'predicted' and prediction['value'] is not None:
            raise ValueError('Unavailable event cannot become absence')
        if any(key not in entries for key in prediction['evidence_ids']):
            raise ValueError('Missing event evidence')
    if envelope['state_candidates'] or envelope['self_report'] is not None or envelope['response'] is not None:
        raise ValueError('ARA-02 must not manufacture states, self reports or responses')
    if any(w['driver_state_usable'] or w['speaker_link']['status'] != 'unknown'
           for w in envelope['person_windows']):
        raise ValueError('Unverified speaker must remain unknown')
    return {'schema_valid': True, 'evidence_count': len(entries), 'observation_count': len(envelope['observations']),
            'evidence_hashes_valid': True, 'evidence_locators_valid': True, 'recognition_truth_evaluated': False}


class ObservationIntegrationSession:
    def __init__(self, config, output_dir, video, audio, *, clients=None):
        self.config = config['observation_integration']
        self.output = Path(output_dir).resolve()
        if not self.output.is_relative_to(WORKSPACE):
            raise ValueError('Output outside workspace')
        self.video = Path(video).resolve()
        self.audio = Path(audio).resolve()
        self.clip_id = self.config.get('clip_id', self.video.stem)
        self.directory = self.output / 'observation_integration'
        self.directory.mkdir(exist_ok=False)
        self.audio_replies = {}
        self.clients = clients
        self.backend_futures = {}
        self.backend_executor = None
        configured_models = {
            'face_detection': config['face_detection']['mediapipe']['model_path'],
            'face_classification': config['emotion_recognition']['original']['model_path'],
            'whisper': config['whisper']['model_path'],
        }
        self.original_models = {k: {'path': str((REPO / p).resolve()), 'sha256': digest(REPO / p)}
                                for k, p in configured_models.items()}
        from modules.media_timeline import probe_video_native
        probe = probe_video_native(self.video, self.config.get('native_probe_backend', 'ffprobe'))
        self.pts = [float(frame['best_effort_timestamp_time']) for frame in probe['frames']]
        self.duration = float(probe['streams'][0]['duration'])
        if not self.pts or self.pts[0] != 0 or any(b <= a for a, b in zip(self.pts, self.pts[1:])):
            raise ValueError('Native PTS required with zero media origin')
        write_json(self.directory / 'native_probe.json', probe)
        original_offset = float(config.get('synchronization', {}).get('audio_offset_seconds', 0))
        if not math.isfinite(original_offset):
            raise ValueError('Finite audio offset required')
        with wave.open(str(self.audio), 'rb') as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
                raise ValueError('Integration requires common mono PCM16 at 16kHz')
            total = stream.getnframes()
            begin = max(0, math.ceil(-original_offset * 16000))
            end = min(total, math.floor((self.duration - original_offset) * 16000))
            if not 0 <= begin < end <= total:
                raise ValueError('No shared audio/video interval')
            stream.setpos(begin)
            pcm = stream.readframes(end - begin)
        self.media_begin = begin / 16000 + original_offset
        self.media_end = end / 16000 + original_offset
        self.samples = end - begin
        self.pcm_sha = hashlib.sha256(pcm).hexdigest()
        self.common_audio = self.directory / 'common_audio.wav'
        with wave.open(str(self.common_audio), 'wb') as stream:
            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            stream.writeframes(pcm)
        config.setdefault('synchronization', {})['audio_offset_seconds'] = self.media_begin
        self.inputs = {'video_sha256': digest(self.video), 'source_audio_sha256': digest(self.audio),
                       'audio_pcm_sha256': self.pcm_sha, 'interval_seconds': [0.0, self.duration],
                       'audio_rate': 16000, 'audio_samples': [begin, end], 'audio_source_samples': total,
                       'audio_original_offset_seconds': original_offset,
                       'common_audio_interval_seconds': [self.media_begin, self.media_end],
                       'common_audio_sample_interval': [0, self.samples],
                       'av_absolute_sync': {'status': 'not_measured', 'error_seconds': None}}
        write_json(self.directory / 'input_manifest.json', self.inputs)
        if self.config.get('parallel_backends', False):
            self.start_backend_tasks()

    def start_backend_tasks(self):
        from concurrent.futures import ThreadPoolExecutor
        import uuid
        self.backend_thread_prefix = 'observations-'+uuid.uuid4().hex
        self.backend_executor = ThreadPoolExecutor(max_workers=3,thread_name_prefix=self.backend_thread_prefix)
        backends = {'visual',*self.config.get('ser_backends',['sensevoice','emotion2vec'])}
        if self.config.get('asr_backend')=='sensevoice':backends.add('sensevoice')
        try:
            for backend in sorted(backends):
                self.backend_futures[backend] = self.backend_executor.submit(self.invoke,backend)
        except BaseException:
            self.close()
            raise

    def close(self):
        executor = getattr(self,'backend_executor',None)
        if executor is not None:
            import threading
            executor.shutdown(wait=True,cancel_futures=True)
            self.backend_executor = None
            write_json(self.directory/'parallel_backends.json',{
                'parallel_backends':True,'tasks':list(self.backend_futures),
                'all_backend_tasks_done':all(f.done() for f in self.backend_futures.values()),
                'owned_threads_stopped':not any(t.name.startswith(self.backend_thread_prefix)
                    and t.is_alive() for t in threading.enumerate())})

    def backend_reply(self,backend):
        future = getattr(self,'backend_futures',{}).get(backend)
        return future.result() if future is not None else self.invoke(backend)

    def invoke(self, backend):
        executable = self.config['visual_python' if backend == 'visual' else 'audio_python']
        script = REPO / 'scripts/observation_worker.py'
        interval = [0.0, self.duration] if backend == 'visual' else [self.media_begin, self.media_end]
        from contextlib import nullcontext
        pooled = self.clients is not None and backend in self.clients
        context = nullcontext(self.clients[backend]) if pooled else SubprocessObservationClient(
            [executable, '-B', script], self.directory / backend, backend_id=backend, allowed_root=WORKSPACE)
        with context as client:
            payload = {'clip_id': self.clip_id}
            if backend == 'visual':
                payload['visual_stride'] = self.config.get('visual_stride', 2)
            else:
                per_backend_threads = self.config.get('audio_torch_threads_by_backend', {})
                thread_count = per_backend_threads.get(backend, self.config.get('audio_torch_threads',
                    2 if self.config.get('parallel_backends',False) else 4))
                if isinstance(thread_count, bool) or not isinstance(thread_count, int) or not 1 <= thread_count <= 6:
                    raise ValueError('audio_torch_threads_out_of_range')
                payload.update(sample_interval=[0, self.samples], pcm_sha256=self.pcm_sha,
                               audio_offset_seconds=self.media_begin, torch_num_threads=thread_count)
            reply = client.submit(self.video if backend == 'visual' else self.common_audio, interval,
                                  timeout_s=self.config.get('worker_timeout_seconds', 120),
                                  mode=self.config.get('worker_mode', 'offline'),
                                  available_until_s=self.duration, **payload).result()
            # Evidence remains self-contained within this window, even though
            # the process and its transport directory span several windows.
            if pooled:
                reply = deepcopy(reply)
                for artifact in reply.get('artifacts', []):
                    source = Path(artifact['resolved_path'])
                    target = self.directory / backend / reply['request_id'] / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as destination, source.open('rb') as original:
                        shutil.copyfileobj(original, destination)
                    if digest(target) != artifact['sha256']:
                        raise ValueError('pooled_artifact_copy_integrity')
                    artifact['resolved_path'] = str(target)
                write_json(self.directory / f'{backend}_client_audit.json', [reply])
            else:
                write_json(self.directory / f'{backend}_client_audit.json', client.audit)
        return reply

    def audio_reply(self, backend):
        if backend not in {'sensevoice', 'emotion2vec'}:
            raise ValueError('Unsupported audio backend')
        if backend not in self.audio_replies:
            self.audio_replies[backend] = self.backend_reply(backend)
        return self.audio_replies[backend]

    def sensevoice_transcript(self):
        reply = self.audio_reply('sensevoice')
        if reply['status'] != 'ok':
            return {'text': '', 'language': 'ja', 'segments': [], 'status': 'inference_failure',
                    'reason': reply['reason'], 'asr_backend': 'sensevoice'}
        raw = read_json(reply['artifacts'][0]['resolved_path'])['observation']
        text = raw['observations'][0]['asr_text']
        return {'text': text, 'language': 'ja', 'status': 'observed', 'asr_backend': 'sensevoice',
                'segment_time_precision': 'whole_input_interval_no_token_times',
                'segments': [{'start': 0.0, 'end': self.samples / 16000, 'text': text,
                              'words': [], 'time_precision': 'whole_input_interval'}] if text else []}

    def finish(self, speech_result, asr_backend):
        visual_reply = self.backend_reply('visual')
        for backend in self.config.get('ser_backends', ['sensevoice', 'emotion2vec']):
            self.audio_reply(backend)
        self.close()
        classification_path = self.output / 'frame_observations.jsonl'
        classification = read_lines(classification_path)
        if len(classification) != len(self.pts):
            raise ValueError('Original app did not process all native frames')
        visual_path = None
        visual_models = None
        visual = []
        if visual_reply['status'] == 'ok':
            summary_path = next(a['resolved_path'] for a in visual_reply['artifacts'] if a['path'].endswith('visual_summary.json'))
            visual_models = read_json(summary_path)['models']
            visual_path = Path(next(a['resolved_path'] for a in visual_reply['artifacts'] if a['path'].endswith('.jsonl')))
            visual = read_lines(visual_path)
            if len(visual) != len(self.pts) or max(abs(r['frame_time_s'] - t) for r, t in zip(visual, self.pts)) > 1e-6:
                raise ValueError('Visual/native timeline mismatch')
        av_reply = None
        if self.config.get('av_sync_enabled', False):
            from modules.av_sync_backend import invoke_file_backend
            av_reply = invoke_file_backend(self.video, self.common_audio, visual_path, classification_path,
                speech_result.get('segments', []), self.media_begin, self.duration, self.directory,
                self.config.get('syncnet_python', self.config['audio_python']),
                self.config.get('worker_timeout_seconds', 120))
        evidence, observations = [], []
        file_hashes = {}

        def register(path, ident, interval, **locator):
            path = Path(path)
            if path not in file_hashes:
                file_hashes[path] = digest(path)
            evidence.append({'evidence_id': ident, 'relative_path': path.relative_to(self.output).as_posix(),
                             'file_sha256': file_hashes[path], 'media_interval_seconds': interval, **locator})
            return ident

        def observation(kind, backend, value, ids, status='observed', quality=None, **extra):
            observations.append({'kind': kind, 'backend_id': backend, 'status': status, 'value': value,
                                 'quality': quality or {'status': 'not_measured', 'value': None},
                                 'evidence_ids': ids, **extra})

        for index, frame in enumerate(classification):
            left = frame['source_pts_seconds']
            right = classification[index + 1]['source_pts_seconds'] if index + 1 < len(classification) else self.duration
            ident = register(classification_path, f'face.{index}', [left, right], line_index=index)
            observation('face_classification', 'fer2013-mini-xception', frame['faces'], [ident],
                        status='observed' if frame['face_count'] == 1 else 'unavailable',
                        score_kind='uncalibrated_raw_scores_and_separate_smoothed_confidence')
        if visual_path:
            for index, row in enumerate(visual):
                right = visual[index + 1]['frame_time_s'] if index + 1 < len(visual) else self.duration
                ident = register(visual_path, f'geometry.{index}', [row['frame_time_s'], right], line_index=index)
                risk = row['eye_hand_risk']
                eye_valid = row['status'] == 'ok' and not risk['hold_eye_scores']
                observation('eye_geometry', 'face_landmarker', row['eye_geometry'] if eye_valid else None, [ident],
                            status='observed' if eye_valid else 'unavailable',
                            quality={'status': 'reported' if eye_valid else 'unavailable', 'value': risk})
                observation('head_transform', 'face_landmarker', row['head_transform'], [ident],
                            status='observed' if row['status'] == 'ok' else 'unavailable')
                mouth = row['hand_mouth']
                observation('mouth_geometry', 'face_landmarker', mouth, [ident],
                            status='unavailable' if mouth['status'] == 'unknown' else 'observed',
                            quality={'status': 'reported', 'value': {'visibility_verified': False,
                                     'projected_hand_risk': mouth['status']}})
                observation('hand_landmarks', 'hand_landmarker', row['hands'], [ident],
                            quality={'status': 'reported', 'value': {'absence_is_not_visibility': True}})
        else:
            for kind in ('eye_geometry', 'head_transform', 'mouth_geometry', 'hand_landmarks'):
                observation(kind, 'visual', None, [], status='inference_failure', failure_reason=visual_reply['reason'])
        speech_path = self.output / 'speech_result.json'
        for index, segment in enumerate(speech_result.get('segments', [])):
            left = max(0.0, segment['start'] + self.media_begin)
            right = min(self.duration, segment['end'] + self.media_begin)
            if right <= left:
                continue
            ident = register(speech_path, f'asr.{index}', [left, right], json_path=['result', 'segments', index])
            observation('asr', asr_backend, segment, [ident], speaker_link={'status': 'unknown', 'track_id': None})
        if not speech_result.get('segments'):
            observation('asr', asr_backend, None, [], status='inference_failure' if speech_result.get('status') == 'inference_failure'
                        else 'unavailable', failure_reason=speech_result.get('reason', 'no_asr_segments'))
        for backend, reply in self.audio_replies.items():
            if reply['status'] != 'ok':
                observation('acoustic_emotion', backend, None, [], status='inference_failure', failure_reason=reply['reason'])
                continue
            path = Path(reply['artifacts'][0]['resolved_path'])
            raw = read_json(path)
            if raw['pcm_sha256'] != self.pcm_sha or raw['sample_interval'] != [0, self.samples]:
                raise ValueError('Audio backend common input mismatch')
            ident = register(path, f'audio.{backend}', [self.media_begin, self.media_end], json_path=['observation'])
            native = raw['observation']
            observation('acoustic_emotion', backend, native['observations'], [ident],
                        quality={'status': 'reported', 'value': native['quality']},
                        score_kind='raw_tags_no_scores' if backend == 'sensevoice' else 'uncalibrated_nine_way_scores',
                        speaker_link={'status': 'unknown', 'track_id': None}, model=native['model'])
        av_model = None
        if av_reply is not None:
            if av_reply['status'] != 'ok':
                observation('av_correspondence', 'syncnet', None, [], status='inference_failure',
                            failure_reason=av_reply['reason'])
            else:
                from modules.av_sync_backend import machine_windows, validate_result
                path = Path(next(a['resolved_path'] for a in av_reply['artifacts']
                                 if a['path'] == 'av_correspondence.json'))
                raw = read_json(path)
                expected_hashes = {'input_path': digest(self.video), 'audio_input_path': digest(self.common_audio),
                                   'geometry_path': digest(visual_path), 'classification_path': digest(classification_path)}
                validate_result(raw, machine_windows(speech_result.get('segments', []), self.media_begin, self.duration), expected_hashes)
                av_model = raw.get('model')
                ident = register(path, 'av_sync.metadata', [0, self.duration])
                observation('av_correspondence_summary', 'syncnet',
                            {'status': raw['status'], 'model_calls': raw['model_calls'],
                             'method_accuracy_validated': False}, [ident])
                for index, record in enumerate(raw['records']):
                    ident = register(path, f'av_sync.{record["segment_index"]}', record['interval_seconds'],
                                     json_path=['records', index])
                    supporting = [ident, f'asr.{record["segment_index"]}']
                    supporting += [f'geometry.{i}' for i in record['geometry_frame_indices']]
                    observation('av_correspondence', 'syncnet', record, supporting,
                                quality={'status': 'reported', 'value': {'scores_calibrated': False,
                                         'visibility_verified': False, 'absolute_sync_measured': False}})
        envelope = {'schema': SCHEMA, 'run_id': self.output.parent.name, 'clip_id': self.clip_id, 'mode': 'offline',
                    'input': self.inputs, 'evidence_registry': evidence, 'observations': observations,
                    'backend_manifest': {'original_models': self.original_models, 'visual_models': visual_models,
                                         'audio_models': {o['backend_id']: o['model'] for o in observations if 'model' in o},
                                         'worker_runtimes': {k: self.config[k] for k in ('visual_python', 'audio_python')}},
                    'person_windows': person_windows(speech_result.get('segments', []), visual, classification,
                                                     offset_s=self.media_begin, duration_s=self.duration),
                    'backend_health': {'visual': visual_reply['status'], **{k: v['status'] for k, v in self.audio_replies.items()}},
                    'event_predictions': {name: {'status': 'not_implemented', 'value': None, 'evidence_ids': []} for name in QUESTIONS},
                    'state_candidates': [], 'state_status': 'not_implemented', 'self_report': None,
                    'preferences': None, 'response': None, 'human_reference_used': False}
        if av_reply is not None:
            envelope['backend_health']['syncnet'] = av_reply['status']
            envelope['backend_manifest']['av_model'] = av_model
            envelope['backend_manifest']['worker_runtimes']['syncnet_python'] = self.config.get('syncnet_python', self.config['audio_python'])
        validation = validate_envelope(envelope, self.output)
        write_json(self.output / 'observation_envelope_v1.json', envelope)
        write_json(self.output / 'observation_validation.json', validation)
        summary = {'schema': SCHEMA, 'clip_id': self.clip_id, 'backend_health': envelope['backend_health'],
                   'observations_by_kind': dict(Counter(o['kind'] for o in observations)),
                   'observation_status_counts': dict(Counter(o['status'] for o in observations)),
                   'person_window_count': len(envelope['person_windows']), 'speaker_verified_count': 0,
                   'state_candidates': [], 'response': None, 'validation': validation,
                   'worker_failures': sum(v != 'ok' for v in envelope['backend_health'].values())}
        write_json(self.output / 'observation_summary.json', summary)
        rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v}</td></tr>' for k, v in summary['observations_by_kind'].items())
        page = '<!doctype html><meta charset="utf-8"><title>観測統合結果</title><h1>観測統合結果</h1>'
        page += f'<p>{html.escape(self.clip_id)}</p><table><tr><th>観測</th><th>件数</th></tr>{rows}</table>'
        page += '<p>話者対応は未確認。状態候補・応答・4項目の認識評価は後続工程です。</p>'
        (self.output / 'observation_summary.html').write_text(page, encoding='utf-8')
        return summary
