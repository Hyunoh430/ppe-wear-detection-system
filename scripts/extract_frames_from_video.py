import cv2
import os

video_path = '/Users/hyunoh/Documents/vscode/embedded/ppe-wear-detection-system/data/v2/nomask2.mp4'
output_folder = '/Users/hyunoh/Documents/vscode/embedded/ppe-wear-detection-system/data/v2/61_nomask'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1초당 1프레임 저장
    if frame_count % int(fps) == 0:
        filename = f"{output_folder}/61_nomask2_{saved_count:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"총 {saved_count}장의 프레임 저장 완료!")
