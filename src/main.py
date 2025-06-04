import cv2
import numpy as np
import time
from picamera2 import Picamera2
import tensorflow as tf

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path="models/best3_float32.tflite")
interpreter.allocate_tensors()

# 입력/출력 텐서 정보
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

# 클래스 이름 정의
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask',
               'with_gloves', 'without_gloves', 'goggles_on', 'goggles_off']

# PiCamera2 초기화
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 320)  # 모델 해상도와 동일
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

while True:
    frame = picam2.capture_array()
    input_tensor = np.expand_dims(frame, axis=0).astype(np.float32)

    # 추론 시작 시간 기록
    start_time = time.time()

    # 추론 실행
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # 추론 종료 시간 기록
    end_time = time.time()
    inference_time = end_time - start_time
    fps = 1.0 / inference_time

    # 출력 결과 가져오기
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (N, 84)

    detected_classes = set()
    for det in output_data:
        conf = sigmoid(det[4])
        if conf < 0.3:
            continue

        cls_scores = sigmoid(det[5:])
        cls_id = np.argmax(cls_scores)
        cls_conf = cls_scores[cls_id]

        if cls_conf * conf < 0.3:
            continue

        detected_classes.add(class_names[cls_id])

    print(f"[FPS: {fps:.2f}] Detected: {', '.join(detected_classes) if detected_classes else 'None'}")

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()
