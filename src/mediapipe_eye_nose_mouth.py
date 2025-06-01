import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,  # 눈동자 추적을 위해 필요
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# 눈 영역 index (Left & Right eye landmarks 일부)
LEFT_EYE_INDEXES = [33, 133]    # 왼쪽 눈 (대략적인 바깥-안쪽 끝점)
RIGHT_EYE_INDEXES = [362, 263]  # 오른쪽 눈

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR → RGB 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # 눈 위치 표시
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            for idx in LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES:
                pt = landmarks.landmark[idx]
                x, y = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Eye Tracking (with Mask)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
