import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
import time

class YOLOv8TFLiteDetector:
    def __init__(self, model_path):
        """TFLite 모델 초기화"""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 입력/출력 텐서 정보 가져오기
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 모델 입력 크기
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
        # 클래스 이름
        self.class_names = [
            'mask_weared_incorrect', 'with_mask', 'without_mask', 
            'with_gloves', 'without_gloves', 'goggles_on'
        ]
        
        # 클래스별 색상 (BGR 형식)
        self.colors = [
            (0, 0, 255),    # mask_weared_incorrect - 빨간색
            (0, 255, 0),    # with_mask - 초록색
            (255, 0, 0),    # without_mask - 파란색
            (0, 255, 255),  # with_gloves - 노란색
            (255, 0, 255),  # without_gloves - 마젠타
            (255, 255, 0)   # goggles_on - 청록색
        ]
        
        print(f"모델 로드 완료 - 입력 크기: {self.input_width}x{self.input_height}")
        print(f"출력 형태: [1, 300, 6] (300개 탐지 결과)")
        print(f"클래스 수: {len(self.class_names)}")
        print("클래스 목록:", self.class_names)
    
    def preprocess_image(self, image):
        """이미지 전처리"""
        # 320x320으로 리사이즈
        resized = cv2.resize(image, (self.input_width, self.input_height))
        # 정규화 (0-255 -> 0-1)
        normalized = resized.astype(np.float32) / 255.0
        # 배치 차원 추가
        input_tensor = np.expand_dims(normalized, axis=0)
        return input_tensor
    
    def postprocess_predictions(self, predictions, original_shape, confidence_threshold=0.5):
        """예측 결과 후처리 - 모델 출력: [1, 300, 6] 형태"""
        detections = []
        
        # 배치 차원 제거: [1, 300, 6] -> [300, 6]
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        original_height, original_width = original_shape[:2]
        
        # 각 탐지 결과 처리 (300개의 탐지 결과)
        for detection in predictions:
            x_center, y_center, width, height, confidence, class_id = detection
            
            # 신뢰도 임계값 확인
            if confidence > confidence_threshold:
                # 정규화된 좌표를 픽셀 좌표로 변환
                # YOLOv8 TFLite는 0-1 범위의 정규화된 좌표를 출력
                x_center_pixel = x_center * original_width
                y_center_pixel = y_center * original_height  
                width_pixel = width * original_width
                height_pixel = height * original_height
                
                # 바운딩 박스 좌표 계산 (x1, y1, x2, y2)
                x1 = int(x_center_pixel - width_pixel / 2)
                y1 = int(y_center_pixel - height_pixel / 2)
                x2 = int(x_center_pixel + width_pixel / 2)
                y2 = int(y_center_pixel + height_pixel / 2)
                
                # 이미지 경계 내로 제한
                x1 = max(0, min(x1, original_width - 1))
                y1 = max(0, min(y1, original_height - 1))
                x2 = max(0, min(x2, original_width - 1))
                y2 = max(0, min(y2, original_height - 1))
                
                class_id = int(class_id)
                if 0 <= class_id < len(self.class_names):
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id]
                    })
        
        return detections
    
    def detect(self, image, confidence_threshold=0.5):
        """객체 탐지 수행"""
        # 전처리
        input_tensor = self.preprocess_image(image)
        
        # 추론
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # 출력 가져오기
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 후처리
        detections = self.postprocess_predictions(predictions, image.shape, confidence_threshold)
        
        return detections
    
    def draw_detections(self, image, detections):
        """탐지 결과를 이미지에 그리기"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_id = detection['class_id']
            class_name = detection['class_name']
            
            # 바운딩 박스 색상
            color = self.colors[class_id % len(self.colors)]
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 텍스트
            label = f"{class_name}: {confidence:.2f}"
            
            # 텍스트 배경 크기 계산
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # 텍스트 배경 그리기
            cv2.rectangle(
                image, 
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color, 
                -1
            )
            
            # 텍스트 그리기
            cv2.putText(
                image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        return image

def main():
    # TFLite 모델 경로 (실제 경로로 변경하세요)
    model_path = "models/best3_float32_v3.tflite"
    
    try:
        # YOLOv8 TFLite 탐지기 초기화
        detector = YOLOv8TFLiteDetector(model_path)
        
        # Picamera2 초기화
        picam2 = Picamera2()
        
        # 카메라 설정
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        print("카메라 시작됨. 'q'를 눌러 종료하세요.")
        
        # FPS 계산을 위한 변수
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            # 프레임 캡처
            frame = picam2.capture_array()
            
            # 객체 탐지
            detections = detector.detect(frame, confidence_threshold=0.5)
            
            # 탐지된 클래스 이름 출력 (탐지가 있을 때만)
            if detections:
                detected_classes = [f"{d['class_name']}({d['confidence']:.2f})" for d in detections]
                print(f"탐지된 객체: {', '.join(detected_classes)}")
            
            # 바운딩 박스 그리기
            annotated_frame = detector.draw_detections(frame.copy(), detections)
            
            # FPS 계산 및 표시
            fps_counter += 1
            if fps_counter % 10 == 0:  # 10프레임마다 FPS 계산
                current_time = time.time()
                fps = 10 / (current_time - fps_start_time)
                print(f"FPS: {fps:.2f}")
                fps_start_time = current_time
            
            # FPS를 이미지에 표시
            cv2.putText(
                annotated_frame, f"Objects: {len(detections)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            # 결과 표시
            cv2.imshow("PPE Detection", annotated_frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except FileNotFoundError:
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        print("모델 파일 경로를 확인하고 다시 시도하세요.")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
    finally:
        # 정리
        try:
            picam2.stop()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()