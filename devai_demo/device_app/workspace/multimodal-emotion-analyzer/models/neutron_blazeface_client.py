"""Client for i.MX95 Neutron NPU BlazeFace service."""
import os
import sys
import time
import socket
import atexit
import subprocess
from pathlib import Path
import cv2
import numpy as np

class NeutronBlazeFaceClient:
    def __init__(self, sock_path="/tmp/neutron_blazeface_v15.sock", model_path=None):
        self.sock_path = sock_path
        self.service_proc = None
        self.sock = None

        if os.path.exists(sock_path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(sock_path)
                self.sock = s
            except Exception:
                try:
                    os.remove(sock_path)
                except Exception:
                    pass

        if self.sock is None:
            service_script = Path(__file__).resolve().parent / "neutron_blazeface_service.py"
            cmd = ["python3", str(service_script), sock_path]
            if model_path:
                cmd.append(str(model_path))

            self.service_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

            t0 = time.monotonic()
            while not os.path.exists(sock_path):
                if time.monotonic() - t0 > 6.0:
                    self.close()
                    raise RuntimeError("Neutron BlazeFace service auto-spawn timed out")
                time.sleep(0.05)

            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(sock_path)
            self.sock = s

        atexit.register(self.close)

    def predict_raw(self, rgb_image: np.ndarray):
        # Expects RGB image, resizes to (128, 128) if needed
        if rgb_image.shape[:2] != (128, 128):
            rgb_128 = cv2.resize(rgb_image, (128, 128))
        else:
            rgb_128 = rgb_image
        if rgb_128.dtype != np.uint8:
            rgb_128 = np.clip(rgb_128, 0, 255).astype(np.uint8)

        self.sock.sendall(rgb_128.tobytes())
        resp = bytearray()
        expected = 14336 + 896  # 15232 bytes
        while len(resp) < expected:
            chunk = self.sock.recv(expected - len(resp))
            if not chunk:
                raise RuntimeError("Neutron BlazeFace socket closed unexpectedly")
            resp.extend(chunk)

        reg_int8 = np.frombuffer(resp[:14336], dtype=np.int8).reshape((1, 896, 16))
        cls_int8 = np.frombuffer(resp[14336:], dtype=np.int8).reshape((1, 896, 1))

        # Dequantize
        # REG: scale 0.84436864, zp -80
        regressors = (reg_int8.astype(np.float32) - (-80)) * 0.84436864
        # CLS: scale 0.6609098, zp 127
        scores = (cls_int8.astype(np.float32) - 127) * 0.6609098
        return regressors, scores

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        if self.service_proc and self.service_proc.poll() is None:
            self.service_proc.terminate()
            try:
                self.service_proc.wait(timeout=2.0)
            except Exception:
                pass
            self.service_proc = None
        if os.path.exists(self.sock_path):
            try:
                os.remove(self.sock_path)
            except Exception:
                pass
