import cv2
import numpy as np
import time
from picamera2 import Picamera2
import tensorflow as tf

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path="models/best3_float32_v3.tflite")
interpreter.allocate_tensors()

# 입력/출력 텐서 정보
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

# 클래스 이름 정의
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask',
               'with_gloves', 'without_gloves', 'goggles_on']

# PiCamera2 초기화
picam2 = Picamera2()
picam2.preview_configuration.main.size = (input_width, input_height)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(1)

# NMS 함수
def non_max_suppression(boxes, scores, iou_threshold=0.4):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=0.5,
        nms_threshold=iou_threshold
    )
    return indices

while True:
    frame = picam2.capture_array()
    resized_frame = cv2.resize(frame, (320, 320))  # 모델 input과 정확히 맞춰야 함
    input_tensor = np.expand_dims(resized_frame, axis=0).astype(np.float32)


    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (N, 6)
    end_time = time.time()
    fps = 1.0 / (end_time - start_time)

    boxes = []
    confidences = []
    class_ids = []

    for det in output_data:
        x, y, w, h, conf, cls_id = det
        if conf < 0.5 or int(cls_id) >= len(class_names):
            continue

        # 정규화 좌표 → 픽셀 변환
        x *= input_width
        y *= input_height
        w *= input_width
        h *= input_height

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x1 = max(0, x1)
        y1 = max(0, y1)

        boxes.append([x1, y1, int(w), int(h)])
        confidences.append(float(conf))
        class_ids.append(int(cls_id))

    indices = non_max_suppression(boxes, confidences)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cls_id = class_ids[i]
            label = f"{class_names[cls_id]} {confidences[i]:.2f}"

            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resized_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(resized_frame, f"FPS: {fps:.2f}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()
