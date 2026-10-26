"""Windows capture adapters. Host receive clocks are never native camera PTS."""
import multiprocessing as mp
import queue
import time


def _camera_worker(device, messages, stopped):
    import cv2
    camera = cv2.VideoCapture(int(device), cv2.CAP_DSHOW)
    try:
        if not camera.isOpened(): raise RuntimeError('windows_camera_open_failed')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        camera.set(cv2.CAP_PROP_FPS,10)
        index,dropped=0,0
        while not stopped.is_set():
            ok,bgr=camera.read()
            stamp=time.monotonic_ns()
            if not ok: raise RuntimeError('windows_camera_read_failed')
            index+=1
            packet={'native_ns':stamp,'available_ns':stamp,'bgr':bgr,
                    'source_frame_index':index,'source_timestamp_ns':None,
                    'timestamp_basis':'host_receive_monotonic_not_camera_native',
                    'capture_transport_dropped_frames':dropped}
            try: messages.put(packet,timeout=.05)
            except queue.Full: dropped+=1
    except BaseException as error:
        try: messages.put({'error':f'{type(error).__name__}: {error}'},timeout=.2)
        except queue.Full:pass
    finally:
        camera.release()
        messages.cancel_join_thread()


def windows_camera_frames(stop,device=0):
    context=mp.get_context('spawn')
    messages=context.Queue(maxsize=2);stopped=context.Event()
    process=context.Process(target=_camera_worker,args=(device,messages,stopped),name='owned_pc_camera')
    process.start()
    try:
        while not stop.is_set():
            try:message=messages.get(timeout=.2)
            except queue.Empty:
                if not process.is_alive():raise RuntimeError('windows_camera_process_exited')
                continue
            if 'error' in message:raise RuntimeError(message['error'])
            yield message
    finally:
        stopped.set();process.join(2)
        if process.is_alive():process.terminate();process.join(3)
        if process.is_alive():process.kill();process.join(3)
        messages.close();messages.join_thread()
        if process.is_alive():raise RuntimeError('windows_camera_not_released')


class AudioClock:
    """Use a causal sample timeline anchored before first callback availability.

    Host APIs can report ADC/current timestamps inconsistent with callback-buffer
    duration. Keep those raw clocks for drift checks, but do not repair them into
    hardware PTS. The 50ms safety lag is explicit application buffering, not an
    estimated device latency or measured absolute AV synchronization.
    """
    def __init__(self,rate=16000):
        self.rate=rate;self.origin=None;self.adc_origin=None;self.samples=0;self.previous_adc=None;self.previous_frames=0

    def map(self,frames,adc,current,available_ns):
        import math
        if not all(math.isfinite(v) for v in (adc,current)) or adc<=0 or adc>current+.002:
            raise ValueError('portaudio_adc_clock_unavailable')
        if self.origin is None:
            self.adc_origin=adc
            self.origin=available_ns-round(frames/self.rate*1e9)-50_000_000
        expected=self.adc_origin+self.samples/self.rate
        drift=adc-expected
        block_drift=0 if self.previous_adc is None else adc-self.previous_adc-self.previous_frames/self.rate
        if abs(block_drift)>.02:raise ValueError('portaudio_sample_clock_gap')
        start=self.origin+round(self.samples/self.rate*1e9)
        end=start+round(frames/self.rate*1e9)
        if end>available_ns+2_000_000:raise ValueError('portaudio_future_samples')
        first=self.samples;self.samples+=frames
        self.previous_adc=adc;self.previous_frames=frames
        return {'sample_interval':[first,self.samples],'native_start_ns':start,
                'available_ns':available_ns,'portaudio_adc_seconds':adc,'portaudio_current_seconds':current,
                'sample_clock_drift_seconds':drift,'timestamp_basis':'callback_availability_sample_timeline_not_native',
                'block_clock_deviation_seconds':block_drift,
                'timeline_safety_lag_ns':50_000_000,'source_timestamp_ns':None,
                'absolute_av_sync_measured':False}


class WindowsMicrophone:
    def __init__(self,device,rate=16000):self.device=device;self.rate=rate

    def run(self,stop,ring,emit):
        import sounddevice as sd
        clock=AudioClock(self.rate);errors=queue.Queue()
        def callback(data,frames,clocks,status):
            if stop.is_set():raise sd.CallbackStop
            try:
                if status:raise RuntimeError('portaudio_status_'+str(status))
                stamp=clock.map(frames,clocks.inputBufferAdcTime,clocks.currentTime,time.monotonic_ns())
                ring.append(stamp['native_start_ns'],bytes(data),stamp['available_ns'])
                emit({'event':'audio_chunk',**stamp})
            except BaseException as error:
                errors.put(error);raise sd.CallbackAbort
        settings=sd.WasapiSettings(auto_convert=True) if sd.query_hostapis(sd.query_devices(self.device)['hostapi'])['name']=='Windows WASAPI' else None
        with sd.RawInputStream(device=self.device,samplerate=self.rate,channels=1,dtype='int16',
                               blocksize=1600,extra_settings=settings,callback=callback) as stream:
            while not stop.wait(.1):
                if not errors.empty():raise errors.get()
                if not stream.active:raise RuntimeError('windows_microphone_stream_stopped')
        if not errors.empty():raise errors.get()


class WindowsDisplay:
    """Use the existing renderer; distinguish software readback from physical HDMI."""
    def __init__(self,output,font_path,*,headless=False):
        from pathlib import Path
        from types import SimpleNamespace as Object
        import numpy as np
        from PIL import ImageFont
        from modules.framebuffer_display import FramebufferDisplay
        self.renderer=FramebufferDisplay.__new__(FramebufferDisplay)
        self.renderer.output=Path(output)
        color=lambda offset:Object(offset=offset,length=8)
        self.renderer.var=Object(xres=1920,yres=1080,xoffset=0,yoffset=0,
                                red=color(16),green=color(8),blue=color(0),transp=Object(length=0))
        self.renderer.rows=np.zeros((1080,1920),dtype=np.uint32)
        self.renderer.fonts={size:ImageFont.truetype(str(font_path),size) for size in (24,32,42,54)}
        self.renderer.frames=0;self.headless=headless;self.title='ARA PC live'
        if not headless:
            import cv2
            cv2.namedWindow(self.title,cv2.WINDOW_NORMAL);cv2.resizeWindow(self.title,1280,720)

    @property
    def frames(self):return self.renderer.frames

    def draw(self,snapshot,*,save=False):
        result=self.renderer.draw(snapshot,save=save)
        result.update(readback_scope='software_render_buffer',physical_screen_readback=False,
                      pc_window_shown=not self.headless)
        if not self.headless:
            import cv2
            cv2.imshow(self.title,self.renderer.rows.view('uint8').reshape(1080,1920,4)[:,:,:3])
            if cv2.waitKey(1)==27 or cv2.getWindowProperty(self.title,cv2.WND_PROP_VISIBLE)<1:
                raise KeyboardInterrupt('pc_display_closed')
        return result

    def close(self):
        if not self.headless:
            import cv2
            cv2.destroyWindow(self.title)
