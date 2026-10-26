import sys
import numpy as np

try:
    print('Testing tf.lite in app_venv...')
    import tensorflow.lite as tflite
    load_del = tflite.experimental.load_delegate
    print('Loading delegate...')
    d = load_del('/usr/lib/libneutron_delegate.so')
    print('Creating interpreter...')
    interp = tflite.Interpreter(model_path='/tmp/npu_test/fer_mini_xception_neutron.tflite', experimental_delegates=[d])
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    data = np.zeros(inp['shape'], dtype=inp['dtype'])
    interp.set_tensor(inp['index'], data)
    print('Invoking...')
    interp.invoke()
    res = interp.get_tensor(out['index'])
    print('app_venv NPU invoke SUCCESS! Output:', res)
except Exception as e:
    import traceback
    traceback.print_exc()
