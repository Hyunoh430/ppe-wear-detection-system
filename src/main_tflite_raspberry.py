import cv2
import numpy as np
import time
from picamera2 import Picamera2
import tensorflow as tf

# 클래스 정의
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask',
               'with_gloves', 'without_gloves', 'goggles_on']

# 모델 로드
interpreter = tf.lite.Interpreter(model_path="models/best3_float32_v3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 크기 고정 (320x320)
input_height, input_width = 320, 320

# PiCamera 설정
picam2 = Picamera2()
picam2.preview_configuration.main.size = (input_width, input_height)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(1)

def non_max_suppression(predictions, iou_threshold=0.4, score_threshold=0.5):
    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        x, y, w, h, conf, cls_id = pred
        if conf < score_threshold:
            continue
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(int(cls_id))

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, iou_threshold)
    result = []
    if len(indices) > 0:
        for i in indices.flatten():
            result.append((scores[i], class_ids[i]))
    return result

while True:
    frame = picam2.capture_array()

    # 프레임을 320x320으로 resize 후 float32로 변환
    resized = cv2.resize(frame, (input_width, input_height))
    input_tensor = np.expand_dims(resized, axis=0).astype(np.float32)

    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    detections = non_max_suppression(output_data)

    end_time = time.time()
    fps = 1.0 / (end_time - start_time)

    detected_classes = {class_names[cls_id] for conf, cls_id in detections}

    print(f"[FPS: {fps:.2f}] Detected: {', '.join(detected_classes) if detected_classes else 'None'}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()
