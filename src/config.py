"""
Configuration settings for the PPE waste disposal system
"""

# Model settings
MODEL_PATH = "models/best3_float32_v3.tflite"
CONFIDENCE_THRESHOLD = 0.3
USE_LETTERBOX_PREPROCESSING = True

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FORMAT = "RGB888"

# Servo motor settings
SERVO_PIN = 2  # BCM GPIO pin
SERVO_FREQUENCY = 50  # Hz
SERVO_MIN_DUTY = 3
SERVO_MAX_DUTY = 12
SERVO_CLOSED_ANGLE = 20   # Door closed position
SERVO_OPEN_ANGLE = 120    # Door open position
SERVO_MOVE_DELAY = 0.1   # Delay between angle steps

# PPE detection settings
REQUIRED_PPE = ["with_mask", "with_gloves", "goggles_on"]
FORBIDDEN_PPE = ["mask_weared_incorrect", "without_mask", "without_gloves"]
PPE_CHECK_DURATION = 3.0  # Seconds to maintain proper PPE before opening
DOOR_OPEN_DURATION = 5.0  # Seconds to keep door open
FPS_UPDATE_INTERVAL = 10  # Update FPS every N frames

# System settings
DEBUG_MODE = True
LOG_DETECTIONS = True