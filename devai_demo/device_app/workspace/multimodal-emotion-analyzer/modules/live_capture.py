"""Bounded native-timestamp capture and causal closed windows (Linux board)."""
from collections import deque
import math
import threading
import time


class ManualCaptureGate:
    """Consume each explicit record command once, using the board monotonic clock."""
    def __init__(self):
        self.pending = None
        self.active = False
        self.seen = set()

    def arm(self, command, now_ns):
        if self.pending is not None or command.get('request_id') in self.seen:
            return
        ident, start = command.get('request_id'), command.get('not_before_ns')
        if (not isinstance(ident,str) or not ident or len(ident)>80
            or isinstance(start,bool) or not isinstance(start,int)
            or not 0 <= start <= now_ns+120_000_000_000 or start < now_ns-30_000_000_000):
            raise ValueError('invalid_or_expired_capture_command')
        self.pending = dict(command)

    def recording(self, native_ns):
        if self.pending is not None and native_ns >= self.pending['not_before_ns']:
            self.active = True
        return self.active

    def finish(self):
        if self.pending is None: raise ValueError('capture_not_armed')
        ident = self.pending['request_id']
        self.seen.add(ident)
        self.pending, self.active = None, False
        return ident


class LatestWindowQueue:
    """One consumer and one latest pending window; superseded intervals are gaps."""
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.gaps = []

    def put(self, window):
        with self.condition:
            if self.closed:
                raise RuntimeError('queue_closed')
            if self.pending is not None:
                self.gaps.append({'window_id': self.pending['window_id'],
                    'interval_ns': self.pending['interval_ns'], 'reason': 'superseded_pending_window'})
            self.pending = window
            self.condition.notify()

    def get(self):
        with self.condition:
            self.condition.wait_for(lambda: self.pending is not None or self.closed)
            if self.closed:
                if self.pending is not None:
                    self.gaps.append({'window_id': self.pending['window_id'],
                        'interval_ns': self.pending['interval_ns'], 'reason': 'stopped_pending_window'})
                    self.pending = None
                return None
            window, self.pending = self.pending, None
            return window

    def close(self):
        with self.condition:
            self.closed = True
            if self.pending is not None:
                self.gaps.append({'window_id': self.pending['window_id'],
                    'interval_ns': self.pending['interval_ns'], 'reason': 'stopped_pending_window'})
                self.pending = None
            self.condition.notify_all()


class SpeechPriorityWindowQueue:
    """Bounded queue retaining the strongest-audio and newest pending windows.

    Closed-window inference is slower than capture on the i.MX95.  A latest-only
    queue can therefore discard the one interval containing speech while the
    consumer is busy.  This queue stays bounded while preserving that candidate.
    """
    def __init__(self, capacity=2):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity != 2:
            raise ValueError('speech_priority_queue_capacity_must_be_two')
        self.capacity = capacity
        self.condition = threading.Condition()
        self.pending = []
        self.closed = False
        self.gaps = []

    @staticmethod
    def _end(window):
        return window['interval_ns'][1]

    @staticmethod
    def _energy(window):
        return float(window.get('audio_rms', 0.0))

    def put(self, window):
        with self.condition:
            if self.closed:
                raise RuntimeError('queue_closed')
            candidates = self.pending + [window]
            strongest = max(candidates, key=lambda item: (self._energy(item), self._end(item)))
            newest = max(candidates, key=self._end)
            selected = [strongest]
            if newest is not strongest:
                selected.append(newest)
            elif len(candidates) > 1:
                selected.append(max((item for item in candidates if item is not strongest), key=self._end))
            retained = {id(item) for item in selected[:self.capacity]}
            for item in candidates:
                if id(item) not in retained:
                    self.gaps.append({'window_id': item['window_id'],
                        'interval_ns': item['interval_ns'], 'audio_rms': self._energy(item),
                        'reason': 'superseded_lower_priority_window'})
            self.pending = selected[:self.capacity]
            self.condition.notify()

    def get(self):
        with self.condition:
            self.condition.wait_for(lambda: self.pending or self.closed)
            if self.closed:
                for item in self.pending:
                    self.gaps.append({'window_id': item['window_id'],
                        'interval_ns': item['interval_ns'], 'audio_rms': self._energy(item),
                        'reason': 'stopped_pending_window'})
                self.pending = []
                return None
            selected = max(self.pending, key=lambda item: (self._energy(item), self._end(item)))
            self.pending.remove(selected)
            return selected

    def close(self):
        with self.condition:
            self.closed = True
            for item in self.pending:
                self.gaps.append({'window_id': item['window_id'],
                    'interval_ns': item['interval_ns'], 'audio_rms': self._energy(item),
                    'reason': 'stopped_pending_window'})
            self.pending = []
            self.condition.notify_all()


class TimedPCM:
    """Retain a fixed duration of PCM with ALSA monotonic trigger/sample clock."""
    def __init__(self, rate=16000, retain_seconds=12):
        self.rate = rate
        self.retain_ns = round(retain_seconds * 1e9)
        self.chunks = deque()
        self.lock = threading.Lock()

    def reset(self):
        """Discard the prior ALSA trigger epoch; never concatenate across a gap."""
        with self.lock:
            self.chunks.clear()

    def append(self, start_ns, pcm, available_ns):
        if len(pcm) % 2 or not pcm:
            raise ValueError('invalid_pcm')
        end_ns = start_ns + round(len(pcm) / 2 / self.rate * 1e9)
        if end_ns > available_ns + 20_000_000:
            raise ValueError('future_audio_timestamp')
        with self.lock:
            if self.chunks and abs(start_ns - self.chunks[-1]['end_ns']) > 125_000:
                raise ValueError('audio_clock_gap')
            self.chunks.append({'start_ns': start_ns, 'end_ns': end_ns,
                                'available_ns': available_ns, 'pcm': pcm})
            while self.chunks and self.chunks[0]['end_ns'] < end_ns - self.retain_ns:
                self.chunks.popleft()

    def slice(self, start_ns, end_ns, available_until_ns):
        if not start_ns < end_ns <= available_until_ns:
            raise ValueError('future_or_invalid_window')
        with self.lock:
            chunks = [c for c in self.chunks if c['end_ns'] > start_ns and c['start_ns'] < end_ns]
            if not chunks or chunks[0]['start_ns'] > start_ns or chunks[-1]['end_ns'] < end_ns:
                raise ValueError('audio_window_not_available')
            pieces = []
            first_ns = None
            for chunk in chunks:
                if chunk['available_ns'] > available_until_ns:
                    raise ValueError('future_audio_availability')
                begin = max(0, math.ceil((start_ns - chunk['start_ns']) * self.rate / 1e9))
                finish = min(len(chunk['pcm']) // 2, math.floor((end_ns - chunk['start_ns']) * self.rate / 1e9))
                if finish > begin:
                    if first_ns is None:
                        first_ns = chunk['start_ns'] + round(begin / self.rate * 1e9)
                    pieces.append(chunk['pcm'][2 * begin:2 * finish])
            pcm = b''.join(pieces)
            if not pcm:
                raise ValueError('empty_audio_window')
            return {'pcm': pcm, 'first_sample_monotonic_ns': first_ns,
                    'sample_count': len(pcm) // 2, 'rate': self.rate}


def validate_closed_window(window, now_ns):
    start, end = window['interval_ns']
    if not 0 <= start < end <= window['closed_available_ns'] <= now_ns:
        raise ValueError('future_or_invalid_closed_window')
    previous = None
    for frame in window['frames']:
        stamp = frame['native_ns']
        if not start <= stamp < end or (previous is not None and stamp <= previous):
            raise ValueError('invalid_native_frame_timeline')
        if frame['available_ns'] > window['closed_available_ns'] or stamp > frame['available_ns']:
            raise ValueError('future_frame')
        previous = stamp
    if not window['frames'] or window['frames'][0]['native_ns'] != start:
        raise ValueError('native_window_origin_required')
    audio = window['audio']
    audio_end = audio['first_sample_monotonic_ns'] + round(audio['sample_count'] / audio['rate'] * 1e9)
    if not start <= audio['first_sample_monotonic_ns'] < audio_end <= end + 1:
        raise ValueError('audio_outside_closed_window')


class NativeMicrophone:
    """Use the system ALSA library, keeping its monotonic hardware timestamps."""
    def __init__(self, device='hw:3,0', rate=16000, idle_recovery_allowed=None,
                 allow_continuous_recovery=False, max_recoveries=100):
        self.device, self.rate = device, rate
        self.idle_recovery_allowed = idle_recovery_allowed
        self.allow_continuous_recovery = allow_continuous_recovery
        self.max_recoveries = max_recoveries

    def run(self, stop, ring, emit):
        import ctypes as c
        lib = c.CDLL('libasound.so.2')
        pointer = c.c_void_p
        class Timespec(c.Structure):
            _fields_ = [('tv_sec', c.c_long), ('tv_nsec', c.c_long)]
        signatures = {
            'snd_pcm_open': ([c.POINTER(pointer), c.c_char_p, c.c_int, c.c_int], c.c_int),
            'snd_pcm_set_params': ([pointer, c.c_int, c.c_int, c.c_uint, c.c_uint, c.c_int, c.c_uint], c.c_int),
            'snd_pcm_readi': ([pointer, pointer, c.c_ulong], c.c_long),
            'snd_pcm_wait': ([pointer, c.c_int], c.c_int),
            'snd_pcm_start': ([pointer], c.c_int),
            'snd_pcm_prepare': ([pointer], c.c_int),
            'snd_pcm_sw_params_malloc': ([c.POINTER(pointer)], c.c_int),
            'snd_pcm_sw_params_current': ([pointer, pointer], c.c_int),
            'snd_pcm_sw_params_set_tstamp_mode': ([pointer, pointer, c.c_int], c.c_int),
            'snd_pcm_sw_params_set_tstamp_type': ([pointer, pointer, c.c_int], c.c_int),
            'snd_pcm_sw_params': ([pointer, pointer], c.c_int),
            'snd_pcm_sw_params_free': ([pointer], None),
            'snd_pcm_status_malloc': ([c.POINTER(pointer)], c.c_int),
            'snd_pcm_status': ([pointer, pointer], c.c_int),
            'snd_pcm_status_get_trigger_htstamp': ([pointer, c.POINTER(Timespec)], None),
            'snd_pcm_status_get_htstamp': ([pointer, c.POINTER(Timespec)], None),
            'snd_pcm_status_get_delay': ([pointer], c.c_long),
            'snd_pcm_status_free': ([pointer], None),
            'snd_pcm_close': ([pointer], c.c_int),
        }
        for name, (args, result) in signatures.items():
            getattr(lib, name).argtypes = args
            getattr(lib, name).restype = result
        def check(code):
            if code < 0:
                raise RuntimeError(f'alsa_error_{code}')
            return code
        pcm, sw, status = pointer(), pointer(), pointer()
        total, trigger_ns = 0, None
        recoveries = 0
        def recover_xrun(code):
            nonlocal total, trigger_ns, recoveries
            if code >= 0:
                return False
            if code != -32:
                check(code)
            continuous = self.allow_continuous_recovery
            if not continuous:
                if (self.idle_recovery_allowed is None or not self.idle_recovery_allowed()):
                    check(code)
                if recoveries >= 3:
                    raise RuntimeError('alsa_idle_xrun_recovery_limit')
            else:
                if recoveries >= self.max_recoveries:
                    raise RuntimeError('alsa_continuous_xrun_recovery_limit')
            previous_trigger = trigger_ns
            check(lib.snd_pcm_prepare(pcm))
            ring.reset()
            total, trigger_ns = 0, None
            check(lib.snd_pcm_start(pcm))
            if not continuous and not self.idle_recovery_allowed():
                raise RuntimeError('alsa_xrun_record_request_race')
            recoveries += 1
            emit({'event': 'audio_xrun_recovery' if continuous else 'audio_idle_recovery',
                  'alsa_error': code,
                  'recovery_count': recoveries, 'previous_trigger_ns': previous_trigger,
                  'previous_pcm_discarded': True, 'new_trigger_epoch_required': True,
                  'available_ns': time.monotonic_ns(),
                  'recording_requested': False,
                  'continuous_mode': continuous})
            return True
        try:
            check(lib.snd_pcm_open(c.byref(pcm), self.device.encode(), 1, 1))
            # S16_LE=2, RW_INTERLEAVED=3, one channel, no resampling.
            check(lib.snd_pcm_set_params(pcm, 2, 3, 1, self.rate, 0, 200000))
            check(lib.snd_pcm_sw_params_malloc(c.byref(sw)))
            check(lib.snd_pcm_sw_params_current(pcm, sw))
            check(lib.snd_pcm_sw_params_set_tstamp_mode(pcm, sw, 1))
            check(lib.snd_pcm_sw_params_set_tstamp_type(pcm, sw, 1))
            check(lib.snd_pcm_sw_params(pcm, sw))
            check(lib.snd_pcm_status_malloc(c.byref(status)))
            check(lib.snd_pcm_start(pcm))
            buffer = c.create_string_buffer(3200)
            while not stop.is_set():
                count = lib.snd_pcm_readi(pcm, buffer, 1600)
                if count == -11:
                    recover_xrun(lib.snd_pcm_wait(pcm, 100))
                    continue
                if recover_xrun(count):
                    continue
                check(count)
                if not count:
                    continue
                if recover_xrun(lib.snd_pcm_status(pcm, status)):
                    continue
                trigger, now = Timespec(), Timespec()
                lib.snd_pcm_status_get_trigger_htstamp(status, c.byref(trigger))
                lib.snd_pcm_status_get_htstamp(status, c.byref(now))
                available = time.monotonic_ns()
                native_trigger = trigger.tv_sec * 10**9 + trigger.tv_nsec
                native_now = now.tv_sec * 10**9 + now.tv_nsec
                if native_trigger <= 0 or abs(native_now - available) > 500_000_000:
                    raise ValueError('alsa_monotonic_timestamp_unavailable')
                if trigger_ns is None:
                    trigger_ns = native_trigger
                if trigger_ns != native_trigger:
                    raise ValueError('alsa_trigger_changed_input_gap')
                start = trigger_ns + round(total / self.rate * 1e9)
                data = buffer.raw[:count * 2]
                ring.append(start, data, available)
                total += count
                emit({'event': 'audio_chunk', 'sample_interval': [total - count, total],
                      'native_start_ns': start, 'native_trigger_ns': trigger_ns,
                      'status_htstamp_ns': native_now, 'delay_frames': lib.snd_pcm_status_get_delay(status),
                      'available_ns': available})
        finally:
            if status: lib.snd_pcm_status_free(status)
            if sw: lib.snd_pcm_sw_params_free(sw)
            if pcm: lib.snd_pcm_close(pcm)


def camera_frames(stop, device='/dev/video4'):
    import av
    previous = None
    with av.open(device, format='v4l2', options={'video_size': '640x480', 'framerate': '10',
                  'input_format': 'mjpeg'}, timeout=(5, 5)) as container:
        for frame in container.decode(video=0):
            if stop.is_set():
                break
            available = time.monotonic_ns()
            if frame.pts is None or frame.time_base is None:
                raise ValueError('camera_native_pts_required')
            native = round(frame.pts * frame.time_base * 10**9)
            if native > available or (previous is not None and native <= previous):
                raise ValueError('invalid_camera_monotonic_pts')
            previous = native
            yield {'native_ns': native, 'pts': frame.pts, 'time_base': str(frame.time_base),
                   'available_ns': available, 'bgr': frame.to_ndarray(format='bgr24')}
    if not stop.is_set():
        raise RuntimeError('camera_stream_ended_without_stop')
