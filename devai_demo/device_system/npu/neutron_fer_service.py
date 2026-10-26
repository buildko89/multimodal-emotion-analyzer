import os
import sys
import time
import socket
import struct
import numpy as np
import tflite_runtime.interpreter as tflite

SOCK_PATH = "/tmp/neutron_fer.sock"
MODEL_PATH = "/tmp/npu_test/fer_mini_xception_neutron.tflite"
DELEGATE_PATH = "/usr/lib/libneutron_delegate.so"

def run_service():
    if os.path.exists(SOCK_PATH):
        os.remove(SOCK_PATH)

    print(f"Loading Neutron Delegate: {DELEGATE_PATH}")
    delegate = tflite.load_delegate(DELEGATE_PATH)
    print(f"Loading Model: {MODEL_PATH}")
    interp = tflite.Interpreter(model_path=MODEL_PATH, experimental_delegates=[delegate])
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
    print("Service warmup complete. Listening on UDS socket...")

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCK_PATH)
    server.listen(1)

    while True:
        conn, _ = server.accept()
        try:
            while True:
                # Expect 4096 bytes (64x64 uint8)
                buf = bytearray()
                while len(buf) < 4096:
                    chunk = conn.recv(4096 - len(buf))
                    if not chunk:
                        break
                    buf.extend(chunk)
                if len(buf) < 4096:
                    break

                face_uint8 = np.frombuffer(buf, dtype=np.uint8).reshape((64, 64))
                # Normalize & quantize to int8
                norm = (face_uint8.astype(np.float32) / 127.5) - 1.0
                q_in = np.clip(np.round(norm / in_scale + in_zp), -128, 127).astype(np.int8)
                q_in = np.reshape(q_in, (1, 64, 64, 1))

                interp.set_tensor(inp['index'], q_in)
                interp.invoke()

                raw_out = interp.get_tensor(out['index'])[0]
                probs = ((raw_out.astype(np.float32) - out_zp) * out_scale).astype(np.float32)
                conn.sendall(probs.tobytes())
        finally:
            conn.close()

if __name__ == "__main__":
    run_service()
