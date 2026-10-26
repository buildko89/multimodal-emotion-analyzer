"""Neutron NPU standalone inference service for FER mini_XCEPTION.
Runs on system python (Python 3.14) with native tflite_runtime and libneutron_delegate.so.
"""
import os
import sys
import socket
import numpy as np
from pathlib import Path
import tflite_runtime.interpreter as tflite

def run_service():
    sock_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/neutron_fer.sock"
    default_model = Path(__file__).resolve().parent / "fer_mini_xception_neutron.tflite"
    model_path = sys.argv[2] if len(sys.argv) > 2 else str(default_model)
    delegate_path = "/usr/lib/libneutron_delegate.so"

    if os.path.exists(sock_path):
        try:
            os.remove(sock_path)
        except Exception:
            pass

    delegate = tflite.load_delegate(delegate_path)
    interp = tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale = inp['quantization_parameters']['scales'][0]
    in_zp = inp['quantization_parameters']['zero_points'][0]
    out_scale = out['quantization_parameters']['scales'][0]
    out_zp = out['quantization_parameters']['zero_points'][0]

    # Warmup
    dummy = np.zeros(inp['shape'], dtype=inp['dtype'])
    interp.set_tensor(inp['index'], dummy)
    interp.invoke()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)

    while True:
        try:
            conn, _ = server.accept()
        except Exception:
            break
        try:
            while True:
                buf = bytearray()
                while len(buf) < 4096:
                    chunk = conn.recv(4096 - len(buf))
                    if not chunk:
                        break
                    buf.extend(chunk)
                if len(buf) < 4096:
                    break

                face_uint8 = np.frombuffer(buf, dtype=np.uint8).reshape((64, 64))
                norm = (face_uint8.astype(np.float32) / 127.5) - 1.0
                q_in = np.clip(np.round(norm / in_scale + in_zp), -128, 127).astype(np.int8)
                q_in = np.reshape(q_in, (1, 64, 64, 1))

                interp.set_tensor(inp['index'], q_in)
                interp.invoke()

                raw_out = interp.get_tensor(out['index'])[0]
                probs = ((raw_out.astype(np.float32) - out_zp) * out_scale).astype(np.float32)
                conn.sendall(probs.tobytes())
        except Exception:
            pass
        finally:
            conn.close()

if __name__ == "__main__":
    run_service()
