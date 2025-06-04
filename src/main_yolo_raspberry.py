from ultralytics import YOLO
import cv2
import time
from picamera2 import Picamera2

# YOLOv8 모델 로드
model = YOLO('models/best3.pt')

# Picamera2 초기화
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  # 또는 모델 입력에 맞게 조정
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(1)  # 카메라 워밍업

while True:
    # 프레임 캡처 (numpy 배열 형태, RGB 포맷)
    frame = picam2.capture_array()

    # 모델 추론 (OpenCV는 BGR, 모델은 RGB이므로 변환 생략 가능)
    results = model(frame)

    # 바운딩 박스 시각화
    annotated_frame = results[0].plot()

    # OpenCV 창으로 출력
    cv2.imshow("YOLOv8 Detection - Picamera2", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 키로 종료
        break

cv2.destroyAllWindows()
picam2.stop()
