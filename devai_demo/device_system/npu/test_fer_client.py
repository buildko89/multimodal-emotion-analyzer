import time
import socket
import numpy as np

SOCK_PATH = "/tmp/neutron_fer.sock"

def test_client():
    print(f"Connecting to {SOCK_PATH} from client...")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCK_PATH)

    labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    dummy_face = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    # 1 trial
    sock.sendall(dummy_face.tobytes())
    res = sock.recv(28)
    probs = np.frombuffer(res, dtype=np.float32)
    idx = int(np.argmax(probs))
    print(f"Single prediction: {labels[idx]} (conf={probs[idx]:.4f}) probs={probs}")

    # Benchmark 100 iterations
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        sock.sendall(dummy_face.tobytes())
        res = sock.recv(28)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / N) * 1000.0
    fps = 1000.0 / avg_ms
    print(f"Client End-to-end Latency (IPC + NPU): {avg_ms:.3f} ms per frame ({fps:.1f} FPS)")
    sock.close()

if __name__ == "__main__":
    test_client()
