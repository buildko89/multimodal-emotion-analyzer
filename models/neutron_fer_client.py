"""Client for i.MX95 Neutron NPU FER mini_XCEPTION service."""
import os
import sys
import time
import socket
import atexit
import subprocess
from pathlib import Path
import numpy as np

class NeutronFERClient:
    def __init__(self, sock_path="/tmp/neutron_fer_v14.sock", model_path=None):
        self.sock_path = sock_path
        self.service_proc = None
        self.sock = None
        self.labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        if os.path.exists(sock_path):
            # Test if existing socket is active
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
            service_script = Path(__file__).resolve().parent / "neutron_fer_service.py"
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
                    raise RuntimeError("Neutron FER service auto-spawn timed out")
                time.sleep(0.05)

            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(sock_path)
            self.sock = s

        atexit.register(self.close)

    def predict(self, face_gray_uint8: np.ndarray):
        # face_gray_uint8 must be shape (64, 64) uint8
        if face_gray_uint8.shape != (64, 64):
            import cv2
            face_gray_uint8 = cv2.resize(face_gray_uint8, (64, 64))
        if face_gray_uint8.dtype != np.uint8:
            face_gray_uint8 = np.clip(face_gray_uint8, 0, 255).astype(np.uint8)

        self.sock.sendall(face_gray_uint8.tobytes())
        resp = bytearray()
        while len(resp) < 28:
            chunk = self.sock.recv(28 - len(resp))
            if not chunk:
                raise RuntimeError("Neutron FER service socket closed unexpectedly")
            resp.extend(chunk)

        probs = np.frombuffer(resp, dtype=np.float32)
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx]), probs

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
