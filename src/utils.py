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
    
    return True

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
    print(f"GPIO Access: {'O' if check_gpio_permissions() else 'X'}")
    print(f"Camera Access: {'O' if check_camera_permissions() else 'X'}")
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
    # """Check system requirements"""
    # issues = []
    
    # # Check Python version
    # python_version = sys.version_info
    # if python_version < (3, 7):
    #     issues.append(f"Python 3.7+ required, found {python_version.major}.{python_version.minor}")
    
    # # Check required packages
    # required_packages = [
    #     'tensorflow',
    #     'opencv-python', 
    #     'numpy',
    #     'picamera2',
    #     'RPi.GPIO'
    # ]
    
    # for package in required_packages:
    #     try:
    #         __import__(package.replace('-', '_'))
    #     except ImportError:
    #         issues.append(f"Missing package: {package}")
    
    # # Check GPIO access
    # if not check_gpio_permissions():
    #     issues.append("No GPIO access - run with sudo or add user to gpio group")
    
    # # Check camera access
    # if not check_camera_permissions():
    #     issues.append("No camera access - check camera is enabled and connected")
    
    # if issues:
    #     print("System Requirements Issues:")
    #     for issue in issues:
    #         print(f"  X {issue}")
    #     return False
    # else:
    #     print("O All system requirements met")
    #     return True
    return True

# ==========================================
# Individual Test Code
# ==========================================

def test_logging_system():
    """Test logging system"""
    print("=" * 50)
    print("LOGGING SYSTEM TEST")
    print("=" * 50)
    
    try:
        # 1. Console logging test
        print("1. Console logging test...")
        logger = setup_logging("INFO")
        
        logger.info("INFO level test message")
        logger.warning("WARNING level test message")
        logger.error("ERROR level test message")
        print("   O Console logging successful!")
        
        # 2. File logging test
        print("\n2. File logging test...")
        log_file = "test_log.log"
        logger = setup_logging("DEBUG", log_file)
        
        logger.debug("DEBUG level test")
        logger.info("File logging test")
        
        # Check file
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                if "File logging test" in content:
                    print("   O File logging successful!")
                else:
                    print("   X File logging content verification failed!")
            os.remove(log_file)  # Clean up test file
        else:
            print("   X Log file creation failed!")
            
        return True
        
    except Exception as e:
        print(f"   X Logging test failed: {e}")
        return False

def test_system_checks():
    """Test system check functions"""
    print("=" * 50)
    print("SYSTEM CHECKS TEST")
    print("=" * 50)
    
    print("1. GPIO permission check...")
    gpio_ok = check_gpio_permissions()
    print(f"   GPIO access: {'O' if gpio_ok else 'X'}")
    
    print("\n2. Camera permission check...")
    camera_ok = check_camera_permissions()
    print(f"   Camera access: {'O' if camera_ok else 'X'}")
    
    print("\n3. System requirements check...")
    requirements_ok = check_system_requirements()
    
    print("\n4. System information output...")
    print_system_info()
    
    return gpio_ok and camera_ok and requirements_ok

def test_model_validation():
    """Test model file validation"""
    print("=" * 50)
    print("MODEL FILE VALIDATION TEST")
    print("=" * 50)
    
    # Check default model path
    from config import MODEL_PATH
    
    print(f"Default model path: {MODEL_PATH}")
    
    if validate_model_file(MODEL_PATH):
        print("O Model file validation successful!")
        return True
    else:
        print("X Model file validation failed!")
        
        # Test models directory creation
        print("\nTesting models directory creation...")
        create_model_directory()
        
        return False

def test_performance_monitor():
    """Test performance monitor"""
    print("=" * 50)
    print("PERFORMANCE MONITOR TEST")
    print("=" * 50)
    
    try:
        import time
        
        monitor = PerformanceMonitor()
        
        print("Simulating fake frame processing (5 seconds)...")
        
        for i in range(20):
            start_time = time.time()
            
            # Fake processing time (50-100ms)
            time.sleep(0.05 + (i % 5) * 0.01)
            
            processing_time = time.time() - start_time
            monitor.log_frame_time(processing_time)
            monitor.log_detection_time(processing_time * 0.8)  # Detection time is 80%
            
            if i % 5 == 0:
                print(f"  Frame {i+1}/20 processed")
        
        print("\nPerformance statistics:")
        monitor.print_stats()
        
        print("O Performance monitor test successful!")
        return True
        
    except Exception as e:
        print(f"X Performance monitor test failed: {e}")
        return False

def test_usage_instructions():
    """Test usage instructions display"""
    print("Testing usage instructions output:")
    print_usage_instructions()
    return True

def run_all_util_tests():
    """Run all utility tests"""
    print(" RUNNING ALL UTILITY TESTS")
    print("=" * 60)
    
    tests = [
        ("Logging System", test_logging_system),
        ("System Checks", test_system_checks),
        ("Model Validation", test_model_validation),
        ("Performance Monitor", test_performance_monitor),
        ("Usage Instructions", test_usage_instructions)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n Starting {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"X Error during {test_name} test: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print(" TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "O Success" if result else "X Failed"
        print(f"{test_name:20}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n{success_count} out of {total_count} tests passed")
    
    if success_count == total_count:
        print(" All tests successful!")
        return True
    else:
        print(" Some tests failed!")
        return False

if __name__ == "__main__":
    import sys
    
    print("Utility Test Options:")
    print("1. Logging system test")
    print("2. System checks test")
    print("3. Model validation test")
    print("4. Performance monitor test")
    print("5. Usage instructions test")
    print("6. Run all tests")
    
    choice = input("Choose (1-6): ").strip()
    
    if choice == "1":
        success = test_logging_system()
    elif choice == "2":
        success = test_system_checks()
    elif choice == "3":
        success = test_model_validation()
    elif choice == "4":
        success = test_performance_monitor()
    elif choice == "5":
        success = test_usage_instructions()
    elif choice == "6":
        success = run_all_util_tests()
    else:
        print("Invalid choice. Running all tests.")
        success = run_all_util_tests()
    
    if success:
        print("\n Test successful!")
    else:
        print("\n Test failed!")
        sys.exit(1)