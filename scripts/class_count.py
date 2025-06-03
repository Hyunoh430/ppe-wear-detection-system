import os
from collections import Counter

# label 폴더 경로 지정 (예: train/labels 또는 test/labels)
label_dir = './v3/train/labels'  # 또는 'test/labels'
#label_dir = './maskdataset/train/labels'  # 또는 'test/labels'

# 클래스 카운터 초기화
class_counter = Counter()

# 라벨 파일 전체 순회
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():  # 빈 줄 방지
                    class_id = line.strip().split()[0]  # 첫 번째 값이 class id
                    class_counter[int(class_id)] += 1

# 결과 출력
print("Class distribution in", label_dir)
for class_id, count in sorted(class_counter.items()):
    print(f"Class {class_id}: {count} instances")
