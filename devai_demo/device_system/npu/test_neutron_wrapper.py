import time
import numpy as np
import tflite_runtime.interpreter as tflite

class NeutronFERClassifier:
    def __init__(self, model_path: str, delegate_path: str = '/usr/lib/libneutron_delegate.so'):
        self.delegate = tflite.load_delegate(delegate_path)
        self.interp = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[self.delegate]
        )
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_scale = self.inp['quantization_parameters']['scales'][0]
        self.in_zp = self.inp['quantization_parameters']['zero_points'][0]
        self.out_scale = self.out['quantization_parameters']['scales'][0]
        self.out_zp = self.out['quantization_parameters']['zero_points'][0]
        self.labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def predict(self, face_gray: np.ndarray):
        # face_gray: (64, 64) uint8 or float32 [-1, 1]
        if face_gray.dtype == np.uint8:
            # uint8 [0..255] -> normalized float [-1..1] -> quantized int8
            norm = (face_gray.astype(np.float32) / 127.5) - 1.0
            q_in = np.clip(np.round(norm / self.in_scale + self.in_zp), -128, 127).astype(np.int8)
        else:
            q_in = np.clip(np.round(face_gray / self.in_scale + self.in_zp), -128, 127).astype(np.int8)
        
        q_in = np.reshape(q_in, (1, 64, 64, 1))
        self.interp.set_tensor(self.inp['index'], q_in)
        self.interp.invoke()
        raw_out = self.interp.get_tensor(self.out['index'])[0]
        probs = (raw_out.astype(np.float32) - self.out_zp) * self.out_scale
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx]), probs

if __name__ == '__main__':
    clf = NeutronFERClassifier('/tmp/npu_test/fer_mini_xception_neutron.tflite')
    dummy_face = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    
    # Warmup
    clf.predict(dummy_face)
    
    # Measure 100 runs
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        label, conf, probs = clf.predict(dummy_face)
    t1 = time.perf_counter()
    
    print(f'NeutronFERClassifier: Predicted={label} Conf={conf:.4f}')
    print(f'Probs={probs}')
    print(f'End-to-end Python method Latency: {(t1 - t0)/N*1000.0:.3f} ms')
