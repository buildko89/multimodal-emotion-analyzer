"""Neutron NPU standalone inference service for BlazeFace.
Runs on system python (Python 3.14) with native tflite_runtime and libneutron_delegate.so.
"""
import os
import sys
import socket
import numpy as np
from pathlib import Path
import tflite_runtime.interpreter as tflite

def run_service():
    sock_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/neutron_blazeface_v15.sock"
    default_model = Path(__file__).resolve().parent / "blazeface_neutron.tflite"
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
    out_reg = interp.get_output_details()[0]
    out_cls = interp.get_output_details()[1]

    # Warmup
    dummy = np.zeros(inp['shape'], dtype=inp['dtype'])
    interp.set_tensor(inp['index'], dummy)
    interp.invoke()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)

    input_bytes = 128 * 128 * 3  # 49152 bytes uint8 RGB
    while True:
        try:
            conn, _ = server.accept()
        except Exception:
            break
        try:
            while True:
                buf = bytearray()
                while len(buf) < input_bytes:
                    chunk = conn.recv(input_bytes - len(buf))
                    if not chunk:
                        break
                    buf.extend(chunk)
                if len(buf) < input_bytes:
                    break

                rgb_uint8 = np.frombuffer(buf, dtype=np.uint8).reshape((1, 128, 128, 3))
                # Quantization: (val / 0.00392157) - 128 = uint8 - 128 = int8
                q_in = (rgb_uint8.astype(np.int16) - 128).astype(np.int8)

                interp.set_tensor(inp['index'], q_in)
                interp.invoke()

                reg_bytes = interp.get_tensor(out_reg['index']).tobytes()  # 14336 bytes
                cls_bytes = interp.get_tensor(out_cls['index']).tobytes()  # 896 bytes
                conn.sendall(reg_bytes + cls_bytes)
        except Exception:
            pass
        finally:
            conn.close()

if __name__ == "__main__":
    run_service()
