from ultralytics import YOLO
import cv2

# 모델 경로 (.pt 파일 경로 입력)
model = YOLO('models/yolov5nu.pt')  # 예: 'runs/detect/train/weights/best.pt'

# 처리할 입력 소스 설정
# 0: 웹캠 / 'path/to/video.mp4': 동영상 / 'path/to/image.jpg': 이미지
source = 0  # 예: webcam, 또는 'sample.jpg', 'video.mp4'

# 비디오 캡처 객체 (웹캠 또는 비디오)
cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 모델 추론
    results = model(frame)

    # 결과에서 바운딩 박스가 적용된 이미지 가져오기
    annotated_frame = results[0].plot()

    # FPS 또는 기타 정보 출력 (선택사항)
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 키 누르면 종료
        break


cap.release()
cv2.destroyAllWindows()