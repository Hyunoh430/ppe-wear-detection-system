import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
import time

class ImprovedYOLOv8TFLiteDetector:
    def __init__(self, model_path):
        """Enhanced TFLite model initialization with better preprocessing"""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output tensor info
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Model input size
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
        # Check input data type and range
        self.input_dtype = self.input_details[0]['dtype']
        print(f"Model input dtype: {self.input_dtype}")
        
        # Class names
        self.class_names = [
            'mask_weared_incorrect', 'with_mask', 'without_mask', 
            'with_gloves', 'without_gloves', 'goggles_on'
        ]
        
        # Colors for each class (BGR format)
        self.colors = [
            (0, 0, 255),    # mask_weared_incorrect - Red
            (0, 255, 0),    # with_mask - Green
            (255, 0, 0),    # without_mask - Blue
            (0, 255, 255),  # with_gloves - Yellow
            (255, 0, 255),  # without_gloves - Magenta
            (255, 255, 0)   # goggles_on - Cyan
        ]
        
        print(f"Model loaded successfully - Input size: {self.input_width}x{self.input_height}")
        print(f"Input data type: {self.input_dtype}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        print(f"Number of classes: {len(self.class_names)}")
    
    def preprocess_image_yolo_style(self, image):
        """YOLOv8 style preprocessing with letterbox and proper normalization"""
        # Get original dimensions
        original_height, original_width = image.shape[:2]
        
        # Calculate scale to fit the input size while maintaining aspect ratio
        scale = min(self.input_width / original_width, self.input_height / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create letterbox (padding)
        top = (self.input_height - new_height) // 2
        bottom = self.input_height - new_height - top
        left = (self.input_width - new_width) // 2
        right = self.input_width - new_width - left
        
        # Add padding with gray color (114, 114, 114) - YOLOv8 default
        letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert BGR to RGB if needed
        if len(letterboxed.shape) == 3 and letterboxed.shape[2] == 3:
            letterboxed = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1 range
        normalized = letterboxed.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_tensor = np.expand_dims(normalized, axis=0)
        
        # Store scale and padding for later use in postprocessing
        self.scale = scale
        self.pad_top = top
        self.pad_left = left
        
        return input_tensor
    
    def preprocess_image_simple(self, image):
        """Simple preprocessing - direct resize (for comparison)"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_image, (self.input_width, self.input_height))
        
        # Normalize to 0-1 range
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_tensor = np.expand_dims(normalized, axis=0)
        
        return input_tensor
    
    def postprocess_predictions_improved(self, predictions, original_shape, confidence_threshold=0.3):
        """Improved post-processing with proper coordinate transformation"""
        detections = []
        
        # Remove batch dimension: [1, 300, 6] -> [300, 6]
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        original_height, original_width = original_shape[:2]
        
        for detection in predictions:
            x_center, y_center, width, height, confidence, class_id = detection
            
            if confidence > confidence_threshold:
                # If using letterbox preprocessing, adjust coordinates
                if hasattr(self, 'scale'):
                    # Remove padding offset
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
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
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
    
    def detect(self, image, confidence_threshold=0.3, use_letterbox=True):
        """Perform object detection with improved preprocessing"""
        # Choose preprocessing method
        if use_letterbox:
            input_tensor = self.preprocess_image_yolo_style(image)
        else:
            input_tensor = self.preprocess_image_simple(image)
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # Get output
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Post-processing
        detections = self.postprocess_predictions_improved(predictions, image.shape, confidence_threshold)
        
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

def test_both_methods(model_path, test_image_path=None):
    """Test both preprocessing methods"""
    detector = ImprovedYOLOv8TFLiteDetector(model_path)
    
    if test_image_path:
        # Test with static image
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"Could not load test image: {test_image_path}")
            return
        
        print("\n=== Testing with Letterbox Preprocessing ===")
        detections_letterbox = detector.detect(image, confidence_threshold=0.3, use_letterbox=True)
        print(f"Detections with letterbox: {len(detections_letterbox)}")
        for det in detections_letterbox:
            print(f"  {det['class_name']}: {det['confidence']:.3f}")
        
        print("\n=== Testing with Simple Preprocessing ===")
        detections_simple = detector.detect(image, confidence_threshold=0.3, use_letterbox=False)
        print(f"Detections with simple resize: {len(detections_simple)}")
        for det in detections_simple:
            print(f"  {det['class_name']}: {det['confidence']:.3f}")
    
    return detector

def main():
    # Model path
    model_path = "models/best3_float32_v3.tflite"
    
    try:
        # Test preprocessing methods first
        detector = test_both_methods(model_path)
        
        # Initialize camera
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        print("\nCamera started. Press 'q' to quit.")
        print("Press 'l' to toggle letterbox preprocessing")
        print("Press '+' to increase confidence threshold")
        print("Press '-' to decrease confidence threshold")
        
        # Variables
        fps_counter = 0
        fps_start_time = time.time()
        use_letterbox = True
        confidence_threshold = 0.3
        
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Object detection
            detections = detector.detect(frame, confidence_threshold=confidence_threshold, use_letterbox=use_letterbox)
            
            # Print detected objects
            if detections:
                detected_classes = [f"{d['class_name']}({d['confidence']:.2f})" for d in detections]
                print(f"Detected: {', '.join(detected_classes)}")
            
            # Draw bounding boxes
            annotated_frame = detector.draw_detections(frame.copy(), detections)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 10 == 0:
                current_time = time.time()
                fps = 10 / (current_time - fps_start_time)
                print(f"FPS: {fps:.2f} | Confidence: {confidence_threshold:.2f} | Letterbox: {use_letterbox}")
                fps_start_time = current_time
            
            # Display info on image
            info_text = f"Objects: {len(detections)} | Conf: {confidence_threshold:.2f} | LB: {use_letterbox}"
            cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show result
            cv2.imshow("PPE Detection - Enhanced", annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                use_letterbox = not use_letterbox
                print(f"Letterbox preprocessing: {use_letterbox}")
            elif key == ord('+') or key == ord('='):
                confidence_threshold = min(0.9, confidence_threshold + 0.05)
                print(f"Confidence threshold: {confidence_threshold:.2f}")
            elif key == ord('-'):
                confidence_threshold = max(0.1, confidence_threshold - 0.05)
                print(f"Confidence threshold: {confidence_threshold:.2f}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        try:
            picam2.stop()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()