import os
import shutil

# 경로 설정
label_path = "./maskdataset2/train/labels"
image_path = "./maskdataset2/train/images"
output_label_dir = "./filtered/train/labels"
output_image_dir = "./filtered/train/images"

# 출력 폴더 생성
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)

# 라벨 파일 반복
for label_file in os.listdir(label_path):
    if not label_file.endswith(".txt"):
        continue

    label_file_path = os.path.join(label_path, label_file)
    with open(label_file_path, 'r') as f:
        lines = f.readlines()

    # 클래스 ID 2만 남김
    class_2_lines = [line for line in lines if line.startswith("2 ")]

    if class_2_lines:
        # 클래스 2 줄만 포함된 라벨 파일 저장
        with open(os.path.join(output_label_dir, label_file), 'w') as f_out:
            f_out.writelines(class_2_lines)

        # 해당 이미지도 복사
        img_file = label_file.replace(".txt", ".jpg")  # 확장자 맞게 조정 필요
        src_img_path = os.path.join(image_path, img_file)
        dst_img_path = os.path.join(output_image_dir, img_file)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
