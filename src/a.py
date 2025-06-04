import tensorflow as tf

# 모델 로드
interpreter = tf.lite.Interpreter(model_path="models/best3_float32.tflite")
interpreter.allocate_tensors()

# 입력/출력 텐서 정보
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("🔍 Output tensor details:")
for detail in output_details:
    print(f"  - Name: {detail['name']}")
    print(f"  - Shape: {detail['shape']}")
    print(f"  - Type: {detail['dtype']}")
