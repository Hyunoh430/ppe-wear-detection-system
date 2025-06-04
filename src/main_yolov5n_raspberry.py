from ultralytics import YOLO
import cv2
import numpy as np
import time
from picamera2 import Picamera2
from libcamera import Transform

# YOLO 모델 로드 (PyTorch .pt 모델)
model = YOLO('models/v5n.pt')  # 예: 'models/yolov8n.pt'

# PiCamera2 초기화
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  # 원하는 해상도 설정
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.transform = Transform(hflip=1, vflip=1)  # 필요 시 반전 설정
picam2.configure("preview")
picam2.start()
time.sleep(1)  # 카메라 준비 시간

# FPS 측정 초기값
prev_time = time.time()
frame_count = 0

while True:
    # 프레임 캡처
    frame = picam2.capture_array()

    # 모델 추론
    results = model(frame)
    annotated_frame = results[0].plot()

    # FPS 계산
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # FPS 표시
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("YOLOv8 + PiCamera2", annotated_frame)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cv2.destroyAllWindows()
picam2.close()
