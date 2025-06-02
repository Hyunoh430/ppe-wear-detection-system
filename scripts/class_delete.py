import os

# 라벨 디렉토리 경로 설정
label_dirs = ['./v3/train/labels', './v3/test/labels']

for label_dir in label_dirs:
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(label_dir, filename)

            # 새로운 라벨 파일 내용 생성 (class 6 제거)
            with open(filepath, 'r') as f:
                lines = f.readlines()

            new_lines = [line for line in lines if not line.startswith('6 ')]

            # 파일 덮어쓰기
            with open(filepath, 'w') as f:
                f.writelines(new_lines)

print("✅ goggles_off (class 6) 라벨 제거 완료!")
