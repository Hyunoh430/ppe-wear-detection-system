"""
PPE Detection module using YOLOv8 TFLite model
Optimized for Raspberry Pi
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple
import logging

from .config import *

class PPEDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        """Initialize PPE detector with TFLite model"""
        self.logger = logging.getLogger(__name__)
        
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input/output tensor info
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Model input size
            self.input_shape = self.input_details[0]['shape']
            self.input_height = self.input_shape[1]
            self.input_width = self.input_shape[2]
            self.input_dtype = self.input_details[0]['dtype']
            
            # Class names
            self.class_names = [
                'mask_weared_incorrect', 'with_mask', 'without_mask', 
                'with_gloves', 'without_gloves', 'goggles_on'
            ]
            
            self.logger.info(f"PPE Detector initialized successfully")
            self.logger.info(f"Model input size: {self.input_width}x{self.input_height}")
            self.logger.info(f"Input data type: {self.input_dtype}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PPE detector: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray, use_letterbox: bool = USE_LETTERBOX_PREPROCESSING) -> np.ndarray:
        """Preprocess image for YOLO inference"""
        if use_letterbox:
            return self._preprocess_letterbox(image)
        else:
            return self._preprocess_simple(image)
    
    def _preprocess_letterbox(self, image: np.ndarray) -> np.ndarray:
        """YOLOv8 style preprocessing with letterbox"""
        original_height, original_width = image.shape[:2]
        
        # Calculate scale to maintain aspect ratio
        scale = min(self.input_width / original_width, self.input_height / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create letterbox padding
        top = (self.input_height - new_height) // 2
        bottom = self.input_height - new_height - top
        left = (self.input_width - new_width) // 2
        right = self.input_width - new_width - left
        
        # Add padding
        letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert BGR to RGB
        if len(letterboxed.shape) == 3 and letterboxed.shape[2] == 3:
            letterboxed = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        
        # Normalize and add batch dimension
        normalized = letterboxed.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        
        # Store preprocessing parameters
        self.scale = scale
        self.pad_top = top
        self.pad_left = left
        
        return input_tensor
    
    def _preprocess_simple(self, image: np.ndarray) -> np.ndarray:
        """Simple preprocessing - direct resize"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_image, (self.input_width, self.input_height))
        
        # Normalize and add batch dimension
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        
        return input_tensor
    
    def postprocess_predictions(self, predictions: np.ndarray, original_shape: Tuple[int, int], 
                              confidence_threshold: float = CONFIDENCE_THRESHOLD) -> List[Dict]:
        """Post-process model predictions"""
        detections = []
        
        # Remove batch dimension
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        original_height, original_width = original_shape[:2]
        
        for detection in predictions:
            x_center, y_center, width, height, confidence, class_id = detection
            
            if confidence > confidence_threshold:
                # Adjust coordinates based on preprocessing method
                if hasattr(self, 'scale'):
                    # Letterbox preprocessing - remove padding and scale
                    x_center = (x_center * self.input_width - self.pad_left) / self.scale
                    y_center = (y_center * self.input_height - self.pad_top) / self.scale
                    width = width * self.input_width / self.scale
                    height = height * self.input_height / self.scale
                else:
                    # Simple scaling
                    x_center = x_center * original_width
                    y_center = y_center * original_height
                    width = width * original_width
                    height = height * original_height
                
                # Calculate bounding box coordinates
                x1 = max(0, min(int(x_center - width / 2), original_width - 1))
                y1 = max(0, min(int(y_center - height / 2), original_height - 1))
                x2 = max(0, min(int(x_center + width / 2), original_width - 1))
                y2 = max(0, min(int(y_center + height / 2), original_height - 1))
                
                class_id = int(class_id)
                if 0 <= class_id < len(self.class_names):
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id]
                    })
        
        return detections
    
    def detect(self, image: np.ndarray, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> List[Dict]:
        """Perform PPE detection on image"""
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            
            # Get predictions
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Post-process
            detections = self.postprocess_predictions(predictions, image.shape, confidence_threshold)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
    
    def check_ppe_compliance(self, detections: List[Dict]) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if all required PPE is properly worn
        Returns: (is_compliant, ppe_status)
        """
        detected_classes = [det['class_name'] for det in detections]
        
        ppe_status = {
            'mask': False,
            'gloves': False, 
            'goggles': False,
            'has_violations': False
        }
        
        # Check for required PPE
        ppe_status['mask'] = 'with_mask' in detected_classes
        ppe_status['gloves'] = 'with_gloves' in detected_classes
        ppe_status['goggles'] = 'goggles_on' in detected_classes
        
        # Check for violations
        violations = any(item in detected_classes for item in FORBIDDEN_PPE)
        ppe_status['has_violations'] = violations
        
        # All required PPE must be present and no violations
        is_compliant = (ppe_status['mask'] and 
                       ppe_status['gloves'] and 
                       ppe_status['goggles'] and 
                       not ppe_status['has_violations'])
        
        return is_compliant, ppe_status
    
    def get_detection_summary(self, detections: List[Dict]) -> str:
        """Get human-readable summary of detections"""
        if not detections:
            return "No PPE detected"
        
        summary = []
        for det in detections:
            summary.append(f"{det['class_name']}({det['confidence']:.2f})")
        
        return ", ".join(summary)


# ==========================================
# 개별 테스트 코드
# ==========================================

def test_ppe_detector():
    """PPE 감지기 개별 테스트"""
    import time
    from picamera2 import Picamera2
    
    print("=" * 50)
    print("PPE DETECTOR 개별 테스트")
    print("=" * 50)
    
    try:
        # 1. 모델 로드 테스트
        print("1. 모델 로드 테스트...")
        detector = PPEDetector()
        print("   ✓ 모델 로드 성공!")
        print(f"   - 입력 크기: {detector.input_width}x{detector.input_height}")
        print(f"   - 클래스 수: {len(detector.class_names)}")
        print(f"   - 클래스 목록: {detector.class_names}")
        
        # 2. 카메라 초기화
        print("\n2. 카메라 초기화...")
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        time.sleep(2)  # 카메라 안정화
        print("   ✓ 카메라 초기화 성공!")
        
        # 3. 실시간 감지 테스트 (10초간)
        print("\n3. 실시간 PPE 감지 테스트 (10초간)...")
        print("   다양한 PPE를 착용하고 테스트해보세요!")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10:
            # 프레임 캡처
            frame = camera.capture_array()
            frame_count += 1
            
            # PPE 감지
            detections = detector.detect(frame)
            
            # 준수 상태 체크
            is_compliant, ppe_status = detector.check_ppe_compliance(detections)
            
            # 결과 출력 (1초마다)
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                
                print(f"\n   [{elapsed:.1f}s] FPS: {fps:.1f}")
                
                if detections:
                    summary = detector.get_detection_summary(detections)
                    print(f"   감지됨: {summary}")
                    
                    print(f"   PPE 상태:")
                    print(f"     - 마스크: {'✓' if ppe_status['mask'] else '✗'}")
                    print(f"     - 장갑: {'✓' if ppe_status['gloves'] else '✗'}")
                    print(f"     - 고글: {'✓' if ppe_status['goggles'] else '✗'}")
                    print(f"     - 위반사항: {'✗' if ppe_status['has_violations'] else '✓'}")
                    print(f"   전체 준수: {'✓ PPE 완전 착용!' if is_compliant else '✗ PPE 미착용/부적절'}")
                else:
                    print("   감지된 PPE 없음")
        
        camera.stop()
        
        # 4. 테스트 완료
        print(f"\n4. 테스트 완료!")
        print(f"   - 총 프레임: {frame_count}")
        print(f"   - 평균 FPS: {frame_count / 10:.1f}")
        print("   ✓ PPE 감지기 정상 작동!")
        
    except Exception as e:
        print(f"   ✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_model_only():
    """모델 로드만 테스트 (카메라 없이)"""
    print("=" * 50)
    print("모델 로드 테스트 (카메라 없이)")
    print("=" * 50)
    
    try:
        detector = PPEDetector()
        print("✓ 모델 로드 성공!")
        print(f"입력 크기: {detector.input_width}x{detector.input_height}")
        print(f"클래스: {detector.class_names}")
        
        # 더미 이미지로 추론 테스트
        import numpy as np
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("\n더미 이미지로 추론 테스트...")
        detections = detector.detect(dummy_image)
        print(f"결과: {len(detections)}개 객체 감지")
        
        return True
        
    except Exception as e:
        print(f"✗ 오류: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    print("PPE Detector 테스트 옵션:")
    print("1. 전체 테스트 (카메라 포함)")
    print("2. 모델만 테스트 (카메라 없이)")
    
    choice = input("선택 (1 또는 2): ").strip()
    
    if choice == "1":
        test_ppe_detector()
    elif choice == "2":
        test_model_only()
    else:
        print("잘못된 선택입니다. 모델만 테스트합니다.")
        test_model_only()