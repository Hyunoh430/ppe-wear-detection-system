"""
Utility functions for the waste disposal system
"""

import logging
import os
import sys
import signal
from datetime import datetime
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if not log_file else [logging.FileHandler(log_file)])
        ]
    )
    
    logger = logging.getLogger("WasteDisposalSystem")
    logger.info(f"Logging initialized - Level: {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return logger

def create_log_filename() -> str:
    """Create timestamped log filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"logs/waste_disposal_{timestamp}.log"

def check_gpio_permissions() -> bool:
    """Check if current user has GPIO permissions"""
    try:
        # Try to access GPIO memory
        with open('/dev/gpiomem', 'r') as f:
            pass
        return True
    except PermissionError:
        return False
    except FileNotFoundError:
        # GPIO not available (probably not on Raspberry Pi)
        return False

def check_camera_permissions() -> bool:
    """Check if camera is available and accessible"""
    try:
        from picamera2 import Picamera2
        
        # Try to create camera instance
        camera = Picamera2()
        camera_info = camera.camera_info
        
        # If we get here, camera is accessible
        return True
        
    except Exception:
        return False

def print_system_info():
    """Print system information"""
    import platform
    import psutil
    
    print("=" * 50)
    print("WASTE DISPOSAL SYSTEM - SYSTEM INFO")
    print("=" * 50)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"CPU Count: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"GPIO Access: {'✓' if check_gpio_permissions() else '✗'}")
    print(f"Camera Access: {'✓' if check_camera_permissions() else '✗'}")
    print("=" * 50)

def signal_handler(system_instance):
    """Create signal handler for graceful shutdown"""
    def handler(signum, frame):
        print(f"\nReceived signal {signum}")
        if system_instance:
            system_instance.stop()
        sys.exit(0)
    
    return handler

def validate_model_file(model_path: str) -> bool:
    """Validate TFLite model file"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    if not model_path.endswith('.tflite'):
        print(f"Warning: Model file should have .tflite extension: {model_path}")
    
    # Check file size (should be reasonable for YOLO model)
    file_size = os.path.getsize(model_path)
    if file_size < 1024 * 1024:  # Less than 1MB
        print(f"Warning: Model file seems too small: {file_size / (1024*1024):.1f} MB")
    elif file_size > 100 * 1024 * 1024:  # More than 100MB
        print(f"Warning: Model file seems very large: {file_size / (1024*1024):.1f} MB")
    else:
        print(f"Model file size: {file_size / (1024*1024):.1f} MB")
    
    return True

def create_model_directory():
    """Create models directory if it doesn't exist"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 60)
    print("WASTE DISPOSAL SYSTEM - USAGE INSTRUCTIONS")
    print("=" * 60)
    print("1. Ensure all PPE is properly worn:")
    print("   - Face mask (properly positioned)")
    print("   - Safety gloves")
    print("   - Safety goggles")
    print("")
    print("2. Stand in front of the camera")
    print("")
    print("3. Maintain proper PPE for 3 seconds")
    print("")
    print("4. Door will open automatically when conditions are met")
    print("")
    print("5. Door will close automatically after 5 seconds")
    print("")
    print("Controls:")
    print("- Ctrl+C: Stop system")
    print("- Emergency stop available via emergency_stop() method")
    print("=" * 60)

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.frame_times = []
        self.detection_times = []
    
    def log_frame_time(self, frame_time: float):
        """Log frame processing time"""
        self.frame_times.append(frame_time)
        
        # Keep only last 100 measurements
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
    
    def log_detection_time(self, detection_time: float):
        """Log detection processing time"""
        self.detection_times.append(detection_time)
        
        # Keep only last 100 measurements
        if len(self.detection_times) > 100:
            self.detection_times.pop(0)
    
    def get_average_frame_time(self) -> float:
        """Get average frame processing time"""
        return sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
    
    def get_average_detection_time(self) -> float:
        """Get average detection time"""
        return sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
    
    def get_fps(self) -> float:
        """Get effective FPS"""
        avg_frame_time = self.get_average_frame_time()
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    def print_stats(self):
        """Print performance statistics"""
        runtime = (datetime.now() - self.start_time).total_seconds()
        avg_frame_time = self.get_average_frame_time()
        avg_detection_time = self.get_average_detection_time()
        
        print(f"\nPerformance Statistics:")
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Average frame time: {avg_frame_time*1000:.1f} ms")
        print(f"Average detection time: {avg_detection_time*1000:.1f} ms")
        print(f"Effective FPS: {self.get_fps():.1f}")

def check_system_requirements():
    """Check system requirements"""
    issues = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 7):
        issues.append(f"Python 3.7+ required, found {python_version.major}.{python_version.minor}")
    
    # Check required packages
    required_packages = [
        'tensorflow',
        'opencv-python', 
        'numpy',
        'picamera2',
        'RPi.GPIO'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            issues.append(f"Missing package: {package}")
    
    # Check GPIO access
    if not check_gpio_permissions():
        issues.append("No GPIO access - run with sudo or add user to gpio group")
    
    # Check camera access
    if not check_camera_permissions():
        issues.append("No camera access - check camera is enabled and connected")
    
    if issues:
        print("System Requirements Issues:")
        for issue in issues:
            print(f"  ✗ {issue}")
        return False
    else:
        print("✓ All system requirements met")
        return True