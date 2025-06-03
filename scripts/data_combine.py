import os
import shutil

# 경로 설정
source_label_dir = './filtered/train/labels'
source_image_dir = './filtered/train/images'

target_label_dir = './v3/train/labels'
target_image_dir = './v3/train/images'

# 클래스 번호 매핑: 'hand 01' → 4, 'hand 02' → 5
class_mapping = {
    '2': '0'
}

# 라벨 수정 및 저장
for label_file in os.listdir(source_label_dir):
    if label_file.endswith('.txt'):
        src_path = os.path.join(source_label_dir, label_file)
        with open(src_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts[0] in class_mapping:
                parts[0] = class_mapping[parts[0]]
                new_lines.append(' '.join(parts) + '\n')

        # 대상 경로에 저장
        os.makedirs(target_label_dir, exist_ok=True)
        with open(os.path.join(target_label_dir, label_file), 'w') as f:
            f.writelines(new_lines)

# 이미지 복사
os.makedirs(target_image_dir, exist_ok=True)
for image_file in os.listdir(source_image_dir):
    if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        shutil.copy2(os.path.join(source_image_dir, image_file),
                     os.path.join(target_image_dir, image_file))

print("✅ hand_dataset 병합 완료 (class 0→4, 1→5 리맵 포함)")
