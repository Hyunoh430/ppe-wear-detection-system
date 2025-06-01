import os
import cv2
import random
from pathlib import Path
import shutil

# 클래스 ID
TARGET_CLASS_ID = 2
# 증강할 수 (2배 → 기존 개수만큼)
AUG_PER_IMAGE = 1

# 경로 설정
images_dir = Path('/Users/hyunoh/Documents/vscode/embedded/ppe-wear-detection-system/data/v1/train/images')
labels_dir = Path('/Users/hyunoh/Documents/vscode/embedded/ppe-wear-detection-system/data/v1/train/labels')
output_images_dir = images_dir
output_labels_dir = labels_dir

# 라벨 파일들 중 클래스 2가 포함된 파일만 필터링
label_files = list(labels_dir.glob('*.txt'))

def contains_target_class(label_path):
    with open(label_path, 'r') as f:
        for line in f:
            if line.strip().startswith(str(TARGET_CLASS_ID) + ' '):
                return True
    return False

# 증강 함수 (간단한 수평 뒤집기 + 밝기 변경)
def augment_image(image):
    # 랜덤 밝기 조절
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = random.uniform(0.7, 1.3)
    hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], value)
    image_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 수평 뒤집기
    flipped = cv2.flip(image_bright, 1)
    return flipped

# 라벨 수평 뒤집기
def flip_yolo_labels(label_path):
    new_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            x_flipped = 1.0 - float(x)
            new_lines.append(f"{cls} {x_flipped:.6f} {y} {w} {h}")
    return new_lines

augmented_count = 0
for label_path in label_files:
    if not contains_target_class(label_path):
        continue

    image_path = images_dir / (label_path.stem + '.jpg')
    if not image_path.exists():
        image_path = images_dir / (label_path.stem + '.png')
        if not image_path.exists():
            continue

    image = cv2.imread(str(image_path))

    for i in range(AUG_PER_IMAGE):
        aug_image = augment_image(image)
        aug_label_lines = flip_yolo_labels(label_path)

        new_stem = f"{label_path.stem}_aug{i}"
        new_image_path = output_images_dir / f"{new_stem}.jpg"
        new_label_path = output_labels_dir / f"{new_stem}.txt"

        cv2.imwrite(str(new_image_path), aug_image)
        with open(new_label_path, 'w') as f:
            f.write('\n'.join(aug_label_lines) + '\n')

        augmented_count += 1

print(f"✅ 클래스 {TARGET_CLASS_ID} 증강 완료: {augmented_count}장 추가됨.")
