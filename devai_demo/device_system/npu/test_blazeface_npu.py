import time
import numpy as np
import tflite_runtime.interpreter as tflite

print('--- Starting i.MX95 Neutron NPU BlazeFace Benchmark ---')
print('Loading Neutron delegate: /usr/lib/libneutron_delegate.so...')
delegate = tflite.load_delegate('/usr/lib/libneutron_delegate.so')

model_path = '/tmp/npu_test/blazeface_neutron.tflite'
print(f'Creating Interpreter for model: {model_path}...')
interp = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[delegate]
)
interp.allocate_tensors()

for i, inp in enumerate(interp.get_input_details()):
    print(f"Input[{i}]: name={inp['name']} shape={inp['shape']} dtype={inp['dtype']}")
for o, out in enumerate(interp.get_output_details()):
    print(f"Output[{o}]: name={out['name']} shape={out['shape']} dtype={out['dtype']}")

inp = interp.get_input_details()[0]
test_data = np.zeros(inp['shape'], dtype=inp['dtype'])
interp.set_tensor(inp['index'], test_data)

# Warmup invoke
print('Executing warmup invocation on Neutron NPU...')
t_warm0 = time.perf_counter()
interp.invoke()
t_warm1 = time.perf_counter()
print(f'Warmup invocation succeeded! Warmup latency: {(t_warm1 - t_warm0)*1000.0:.3f} ms')

# Benchmark iterations
iters = 50
print(f'Benchmarking {iters} iterations on Neutron NPU...')
t0 = time.perf_counter()
for _ in range(iters):
    interp.set_tensor(inp['index'], test_data)
    interp.invoke()
t1 = time.perf_counter()

avg_ms = ((t1 - t0) / iters) * 1000.0
fps = 1000.0 / avg_ms
print(f'Average NPU Latency: {avg_ms:.3f} ms per inference ({fps:.1f} inferences/sec)')
print('--- BlazeFace Benchmark complete! ---')
