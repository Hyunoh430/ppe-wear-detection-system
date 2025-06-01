import os
from pathlib import Path

def check_yolo_dataset_matching(images_dir, labels_dir):
    image_files = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
    missing_labels = []
    matched = 0

    for image_path in image_files:
        label_path = Path(labels_dir) / (image_path.stem + '.txt')
        if not label_path.exists():
            missing_labels.append(str(image_path))
        else:
            matched += 1

    print(f"총 이미지 수: {len(image_files)}")
    print(f"매칭된 라벨 수: {matched}")
    print(f"라벨이 없는 이미지 수: {len(missing_labels)}")

    if missing_labels:
        print("\n⚠️ 라벨이 없는 이미지 목록:")
        for path in missing_labels:
            print(f" - {path}")

# 디렉토리 경로 설정
train_images = '/Users/hyunoh/Documents/vscode/embedded/ppe-wear-detection-system/data/v1/train/images'
train_labels = '/Users/hyunoh/Documents/vscode/embedded/ppe-wear-detection-system/data/v1/train/labels'

test_images = '/Users/hyunoh/Documents/vscode/embedded/ppe-wear-detection-system/data/v1/test/images'
test_labels = '/Users/hyunoh/Documents/vscode/embedded/ppe-wear-detection-system/data/v1/test/labels'

print("🔎 [TRAIN 데이터셋 검사]")
check_yolo_dataset_matching(train_images, train_labels)
print("\n🔎 [TEST 데이터셋 검사]")
check_yolo_dataset_matching(test_images, test_labels)
