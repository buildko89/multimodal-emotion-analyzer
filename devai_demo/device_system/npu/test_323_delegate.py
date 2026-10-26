import tflite_runtime.interpreter as tflite
import numpy as np

print('Testing 3.2.3 delegate...')
d = tflite.load_delegate('/tmp/npu_test/libneutron_delegate_323.so')
interp = tflite.Interpreter(
    model_path='/tmp/npu_test/fer_mini_xception_neutron.tflite',
    experimental_delegates=[d]
)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
interp.set_tensor(inp['index'], np.zeros(inp['shape'], dtype=inp['dtype']))
interp.invoke()
print('3.2.3 delegate SUCCESS on system python! Output:', interp.get_tensor(out['index']))
