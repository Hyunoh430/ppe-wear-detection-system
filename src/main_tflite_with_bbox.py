import cv2
import numpy as np
import time
from picamera2 import Picamera2
import tensorflow as tf

# 클래스 이름 정의
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask',
               'with_gloves', 'without_gloves', 'goggles_on']

# TFLite 모델 로드
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

# NMS 함수
def non_max_suppression(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=0.25,
        nms_threshold=iou_threshold
    )
    return indices

while True:
    frame = picam2.capture_array()
    resized_frame = cv2.resize(frame, (input_width, input_height))
    input_tensor = np.expand_dims(resized_frame / 255.0, axis=0).astype(np.float32)

    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    end_time = time.time()
    fps = 1.0 / (end_time - start_time + 1e-6)

    boxes = []
    confidences = []
    class_ids = []

    for det in output_data:
        if len(det) < 6:
            continue

        x_center, y_center, w, h, conf, cls_id = det

        # 1. 전체 객체 신뢰도 (obj_conf * cls_conf) 가 0.25 미만이면 무시
        if conf < 0.25:
            continue
        if int(cls_id) >= len(class_names):
            continue

        # 2. 디코딩: YOLOv8 구조와 동일하게 중심좌표 → 좌상단 기준으로 변환
        x1 = int((x_center - w / 2) * input_width)
        y1 = int((y_center - h / 2) * input_height)
        x2 = int((x_center + w / 2) * input_width)
        y2 = int((y_center + h / 2) * input_height)

        # 3. Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(input_width - 1, x2), min(input_height - 1, y2)
        w_pixel = x2 - x1
        h_pixel = y2 - y1

        # 4. 너무 작은 박스 무시 (노이즈 제거)
        if w_pixel < 5 or h_pixel < 5:
            continue

        boxes.append([x1, y1, w_pixel, h_pixel])
        confidences.append(float(conf))
        class_ids.append(int(cls_id))

    indices = non_max_suppression(boxes, confidences)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{class_names[class_ids[i]]} {confidences[i]:.2f}"

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
