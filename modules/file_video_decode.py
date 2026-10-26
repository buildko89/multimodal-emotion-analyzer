"""Explicit deterministic file color conversion; native capture remains default."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

BACKEND='pyav-bitexact-rgb-v1'
MAX_FRAME_BYTES=64*1024**2


def exact_rgb(frame):
    from av.video.reformatter import Interpolation
    flags=Interpolation.BILINEAR|Interpolation.ACCURATE_RND|Interpolation.BITEXACT
    return frame.to_ndarray(format='rgb24',interpolation=flags)


def frames(path):
    import av
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        fps=float(stream.average_rate) if stream.average_rate else 0
        if not math.isfinite(fps) or fps<=0:raise ValueError('file_decoder_invalid_fps')
        yield {'fps':fps,'width':stream.width,'height':stream.height,'frames':stream.frames}
        last=-1.0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:raise ValueError('file_decoder_missing_pts')
            pts=float(frame.pts*frame.time_base)
            if not math.isfinite(pts) or pts<=last:raise ValueError('file_decoder_nonincreasing_pts')
            last=pts
            yield pts,exact_rgb(frame)


class BitexactCapture:
    def __init__(self,path,decoder_python=None):
        self.closed=False;self.pts=0.;self.index=0;self.proc=None;self.iterator=None
        try:
            if decoder_python and Path(decoder_python).resolve()!=Path(sys.executable).resolve():
                self.proc=subprocess.Popen([str(decoder_python),'-B',str(Path(__file__).resolve()),'--stream',str(path)],
                    stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                header=self.proc.stdout.readline(4096)
                if not header or len(header)>4095:raise ValueError('file_decoder_missing_header')
                self.metadata=json.loads(header)
            else:
                self.iterator=frames(path);self.metadata=next(self.iterator)
        except BaseException:
            self.release();raise

    def isOpened(self):return not self.closed

    def get(self,key):
        import cv2
        values={cv2.CAP_PROP_FPS:self.metadata['fps'],cv2.CAP_PROP_FRAME_WIDTH:self.metadata['width'],
            cv2.CAP_PROP_FRAME_HEIGHT:self.metadata['height'],cv2.CAP_PROP_FRAME_COUNT:self.metadata['frames'],
            cv2.CAP_PROP_POS_MSEC:self.pts*1000,cv2.CAP_PROP_POS_FRAMES:self.index}
        return values.get(key,0)

    def _read_exact(self,length):
        chunks=[];remaining=length
        while remaining:
            part=self.proc.stdout.read(remaining)
            if not part:raise ValueError('file_decoder_truncated_frame')
            chunks.append(part);remaining-=len(part)
        return b''.join(chunks)

    def read(self):
        import numpy as np
        if self.closed:return False,None
        if self.proc:
            header=self._read_exact(20)
            pts,height,width=struct.unpack('<dII',header[:16]);length=struct.unpack('<I',header[16:])[0]
            if length==0:
                code=self.proc.wait(timeout=10)
                if code:raise RuntimeError('file_decoder_subprocess_failed')
                return False,None
            if not 0<length<=MAX_FRAME_BYTES or length!=height*width*3:raise ValueError('file_decoder_frame_size')
            rgb=np.frombuffer(self._read_exact(length),dtype=np.uint8).reshape(height,width,3)
        else:
            try:pts,rgb=next(self.iterator)
            except StopIteration:return False,None
        self.pts=pts;self.index+=1
        return True,np.ascontiguousarray(rgb[:,:,::-1])

    def release(self):
        self.closed=True
        if self.iterator is not None:self.iterator.close();self.iterator=None
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=5)
            self.proc.stdout.close();self.proc.stderr.close()


def open_video(source,backend='opencv-native',decoder_python=None):
    import cv2
    if backend=='opencv-native' or not isinstance(source,(str,Path)):
        return cv2.VideoCapture(source)
    if backend!=BACKEND:raise ValueError('unsupported_file_video_decoder')
    return BitexactCapture(source,decoder_python)


def stream(path):
    iterator=frames(path);output=sys.stdout.buffer
    output.write((json.dumps(next(iterator))+'\n').encode());output.flush()
    for pts,rgb in iterator:
        data=rgb.tobytes();h,w=rgb.shape[:2]
        if len(data)>MAX_FRAME_BYTES:raise ValueError('file_decoder_frame_size')
        output.write(struct.pack('<dIII',pts,h,w,len(data)));output.write(data);output.flush()
    output.write(struct.pack('<dIII',0,0,0,0));output.flush()


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--stream':raise SystemExit('Expected --stream saved_video')
    stream(sys.argv[2])
