import os
import sys
import time
import socket
import subprocess
from pathlib import Path
import numpy as np

SOCK_PATH = "/tmp/neutron_fer_autospawn.sock"
SERVICE_SCRIPT = "/tmp/npu_test/neutron_fer_service_standalone.py"

class NeutronFERClient:
    def __init__(self, sock_path=SOCK_PATH, service_script=SERVICE_SCRIPT):
        self.sock_path = sock_path
        self.service_proc = None
        if os.path.exists(sock_path):
            try:
                os.remove(sock_path)
            except Exception:
                pass

        print(f"Spawning NPU service: {service_script}...")
        self.service_proc = subprocess.Popen(
            ["python3", service_script, sock_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

        t0 = time.monotonic()
        while not os.path.exists(sock_path):
            if time.monotonic() - t0 > 5.0:
                raise RuntimeError("Service auto-spawn timed out")
            time.sleep(0.05)
        print(f"Service ready in {time.monotonic() - t0:.3f} seconds!")

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(sock_path)
        self.labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def predict(self, face_gray_64x64):
        self.sock.sendall(face_gray_64x64.tobytes())
        resp = self.sock.recv(28)
        probs = np.frombuffer(resp, dtype=np.float32)
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx]), probs

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
        if self.service_proc and self.service_proc.poll() is None:
            self.service_proc.terminate()
            self.service_proc.wait(timeout=2.0)
            self.service_proc = None
        if os.path.exists(self.sock_path):
            try:
                os.remove(self.sock_path)
            except Exception:
                pass
        print("NeutronFERClient cleaned up.")

if __name__ == "__main__":
    client = NeutronFERClient()
    try:
        dummy_face = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        label, conf, probs = client.predict(dummy_face)
        print(f"Prediction: {label} (conf={conf:.4f})")
        
        # Benchmark 50 runs
        N = 50
        t0 = time.perf_counter()
        for _ in range(N):
            client.predict(dummy_face)
        t1 = time.perf_counter()
        print(f"Latency: {(t1-t0)/N*1000.0:.3f} ms/frame")
    finally:
        client.close()
