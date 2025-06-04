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
        
        print(f"Model loaded successfully - Input size: {self.input_width}x{self.input_height}")
        print(f"Output shape: [1, 300, 6] (300 detection results)")
        print(f"Number of classes: {len(self.class_names)}")
        print("Class names:", self.class_names)
    
    def preprocess_image(self, image):
        """Image preprocessing"""
        # Resize to 320x320
        resized = cv2.resize(image, (self.input_width, self.input_height))
        # Normalize (0-255 -> 0-1)
        normalized = resized.astype(np.float32) / 255.0
        # Add batch dimension
        input_tensor = np.expand_dims(normalized, axis=0)
        return input_tensor
    
    def postprocess_predictions(self, predictions, original_shape, confidence_threshold=0.5):
        """Post-process predictions - Model output: [1, 300, 6] format"""
        detections = []
        
        # Remove batch dimension: [1, 300, 6] -> [300, 6]
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        original_height, original_width = original_shape[:2]
        
        # Process each detection result (300 detection results)
        for detection in predictions:
            x_center, y_center, width, height, confidence, class_id = detection
            
            # Check confidence threshold
            if confidence > confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                # YOLOv8 TFLite outputs normalized coordinates in 0-1 range
                x_center_pixel = x_center * original_width
                y_center_pixel = y_center * original_height  
                width_pixel = width * original_width
                height_pixel = height * original_height
                
                # Calculate bounding box coordinates (x1, y1, x2, y2)
                x1 = int(x_center_pixel - width_pixel / 2)
                y1 = int(y_center_pixel - height_pixel / 2)
                x2 = int(x_center_pixel + width_pixel / 2)
                y2 = int(y_center_pixel + height_pixel / 2)
                
                # Clip to image boundaries
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
        """Perform object detection"""
        # Preprocessing
        input_tensor = self.preprocess_image(image)
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # Get output
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Post-processing
        detections = self.postprocess_predictions(predictions, image.shape, confidence_threshold)
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw detection results on image"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_id = detection['class_id']
            class_name = detection['class_name']
            
            # Bounding box color
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text background size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw text background
            cv2.rectangle(
                image, 
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        return image

def main():
    # TFLite model path (change to your actual path)
    model_path = "models/best3_float32_v3.tflite"
    
    try:
        # Initialize YOLOv8 TFLite detector
        detector = YOLOv8TFLiteDetector(model_path)
        
        # Initialize Picamera2
        picam2 = Picamera2()
        
        # Camera configuration
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        print("Camera started. Press 'q' to quit.")
        
        # Variables for FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Object detection
            detections = detector.detect(frame, confidence_threshold=0.5)
            
            # Print detected class names (only when there are detections)
            if detections:
                detected_classes = [f"{d['class_name']}({d['confidence']:.2f})" for d in detections]
                print(f"Detected objects: {', '.join(detected_classes)}")
            
            # Draw bounding boxes
            annotated_frame = detector.draw_detections(frame.copy(), detections)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 10 == 0:  # Calculate FPS every 10 frames
                current_time = time.time()
                fps = 10 / (current_time - fps_start_time)
                print(f"FPS: {fps:.2f}")
                fps_start_time = current_time
            
            # Display FPS on image
            cv2.putText(
                annotated_frame, f"Objects: {len(detections)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            # Show result
            cv2.imshow("PPE Detection", annotated_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please check the model file path and try again.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        # Cleanup
        try:
            picam2.stop()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()