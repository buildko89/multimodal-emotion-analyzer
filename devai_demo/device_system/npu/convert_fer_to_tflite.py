import os
import tensorflow as tf
from pathlib import Path

def convert_h5_to_tflite():
    src_h5 = Path(r"D:\source\repos\ara\multimodal-emotion-analyzer\trained_models\emotion_models\fer2013_mini_XCEPTION.102-0.66.hdf5")
    out_dir = Path(r"D:\source\repos\ara\devai\devai_ara\npu_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp32 = out_dir / "fer_mini_xception_fp32.tflite"

    print(f"Loading Keras model from {src_h5}...")
    model = tf.keras.models.load_model(str(src_h5), compile=False)
    model.summary()

    # Create concrete function with fixed batch size 1
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, 64, 64, 1], model.inputs[0].dtype)
    )

    print("Converting to TFLite FP32...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    with open(out_fp32, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite FP32 to {out_fp32} ({len(tflite_model)} bytes)")

if __name__ == "__main__":
    convert_h5_to_tflite()
