import os
import cv2

# 디렉토리 설정
image_dir = './train/images'   # 또는 './test/images'
label_dir = './train/labels'   # 또는 './test/labels'
output_dir = './output_visualized'  # 저장 폴더

os.makedirs(output_dir, exist_ok=True)

# 대상 클래스
target_classes = [0, 1, 2]
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask', 
               'with_gloves', 'without_gloves', 'goggles_on', 'goggles_off']

def yolo_to_bbox(x_center, y_center, w, h, img_w, img_h):
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)
    return x1, y1, x2, y2

# 라벨 순회
for filename in os.listdir(label_dir):
    if not filename.endswith('.txt'):
        continue

    label_path = os.path.join(label_dir, filename)
    image_path_jpg = os.path.join(image_dir, filename.replace('.txt', '.jpg'))
    image_path_png = os.path.join(image_dir, filename.replace('.txt', '.png'))

    image_path = image_path_jpg if os.path.exists(image_path_jpg) else image_path_png
    if not os.path.exists(image_path):
        continue

    image = cv2.imread(image_path)
    if image is None:
        continue

    img_h, img_w = image.shape[:2]
    found = False

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls, x, y, w, h = int(parts[0]), *map(float, parts[1:])
            if cls in target_classes:
                x1, y1, x2, y2 = yolo_to_bbox(x, y, w, h, img_w, img_h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, class_names[cls], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                found = True

    if found:
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
