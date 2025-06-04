import cv2
import numpy as np
import time
from picamera2 import Picamera2
import tensorflow as tf

# 클래스 이름 정의
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask',
               'with_gloves', 'without_gloves', 'goggles_on']

# TFLite 모델 로드 (NMS 포함 모델)
interpreter = tf.lite.Interpreter(model_path="models/best3_float32_v3.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

# PiCamera2 설정
picam2 = Picamera2()
picam2.preview_configuration.main.size = (input_width, input_height)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(1)

while True:
    frame = picam2.capture_array()
    resized_frame = cv2.resize(frame, (input_width, input_height))
    input_tensor = np.expand_dims(resized_frame / 255.0, axis=0).astype(np.float32)

    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    end_time = time.time()
    fps = 1.0 / (end_time - start_time + 1e-6)

    # 출력 텐서 가져오기
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]      # [N, 4]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]    # [N]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]     # [N]
    num = int(interpreter.get_tensor(output_details[3]['index'])[0])   # []

    for i in range(num):
        score = scores[i]
        if score < 0.3:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        cls_id = int(classes[i])

        x1 = int(xmin * input_width)
        y1 = int(ymin * input_height)
        x2 = int(xmax * input_width)
        y2 = int(ymax * input_height)

        label = f"{class_names[cls_id]} {score:.2f}"
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(resized_frame, f"FPS: {fps:.2f}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()
