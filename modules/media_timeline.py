"""Native recording timestamps from an explicit ffprobe or PyAV backend."""
import json
import math
import subprocess


def probe_video_native(path, backend='ffprobe'):
    if backend=='ffprobe':
        return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_frames','-show_streams','-show_entries','stream=duration:frame=best_effort_timestamp_time',
            '-of','json',str(path)]))
    if backend!='pyav':
        raise ValueError('Unsupported native timestamp backend')
    import av
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('No video stream')
        stream=container.streams.video[0]
        if stream.duration is None or stream.time_base is None:
            raise ValueError('Native stream duration required; no FPS-derived fallback')
        duration=float(stream.duration*stream.time_base)
        if not math.isfinite(duration) or duration<=0:
            raise ValueError('Invalid native stream duration')
        frames=[]
        previous=None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Native frame PTS required')
            stamp=float(frame.pts*frame.time_base)
            if not math.isfinite(stamp) or stamp<0 or (previous is not None and stamp<=previous) or stamp>=duration:
                raise ValueError('Invalid native frame timeline')
            previous=stamp
            frames.append({'best_effort_timestamp_time':str(stamp)})
        if not frames:
            raise ValueError('No video frames')
        return {'frames':frames,'streams':[{'duration':str(duration)}],
                'probe_backend':'pyav','runtime_version':av.__version__}
