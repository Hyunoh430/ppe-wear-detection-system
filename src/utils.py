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


# ==========================================
# 개별 테스트 코드
# ==========================================

def test_logging_system():
    """로깅 시스템 테스트"""
    print("=" * 50)
    print("로깅 시스템 테스트")
    print("=" * 50)
    
    try:
        # 1. 콘솔 로깅 테스트
        print("1. 콘솔 로깅 테스트...")
        logger = setup_logging("INFO")
        
        logger.info("INFO 레벨 테스트 메시지")
        logger.warning("WARNING 레벨 테스트 메시지")
        logger.error("ERROR 레벨 테스트 메시지")
        print("   ✓ 콘솔 로깅 성공!")
        
        # 2. 파일 로깅 테스트
        print("\n2. 파일 로깅 테스트...")
        log_file = "test_log.log"
        logger = setup_logging("DEBUG", log_file)
        
        logger.debug("DEBUG 레벨 테스트")
        logger.info("파일 로깅 테스트")
        
        # 파일 확인
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                if "파일 로깅 테스트" in content:
                    print("   ✓ 파일 로깅 성공!")
                else:
                    print("   ✗ 파일 로깅 내용 확인 실패!")
            os.remove(log_file)  # 테스트 파일 정리
        else:
            print("   ✗ 로그 파일 생성 실패!")
            
        return True
        
    except Exception as e:
        print(f"   ✗ 로깅 테스트 실패: {e}")
        return False

def test_system_checks():
    """시스템 체크 함수들 테스트"""
    print("=" * 50)
    print("시스템 체크 테스트")
    print("=" * 50)
    
    print("1. GPIO 권한 체크...")
    gpio_ok = check_gpio_permissions()
    print(f"   GPIO 접근: {'✓' if gpio_ok else '✗'}")
    
    print("\n2. 카메라 권한 체크...")
    camera_ok = check_camera_permissions()
    print(f"   카메라 접근: {'✓' if camera_ok else '✗'}")
    
    print("\n3. 시스템 요구사항 체크...")
    requirements_ok = check_system_requirements()
    
    print("\n4. 시스템 정보 출력...")
    print_system_info()
    
    return gpio_ok and camera_ok and requirements_ok

def test_model_validation():
    """모델 파일 검증 테스트"""
    print("=" * 50)
    print("모델 파일 검증 테스트")
    print("=" * 50)
    
    # 기본 모델 경로 확인
    from .config import MODEL_PATH
    
    print(f"기본 모델 경로: {MODEL_PATH}")
    
    if validate_model_file(MODEL_PATH):
        print("✓ 모델 파일 검증 성공!")
        return True
    else:
        print("✗ 모델 파일 검증 실패!")
        
        # models 디렉토리 생성 테스트
        print("\nmodels 디렉토리 생성 테스트...")
        create_model_directory()
        
        return False

def test_performance_monitor():
    """성능 모니터 테스트"""
    print("=" * 50)
    print("성능 모니터 테스트")
    print("=" * 50)
    
    try:
        import time
        
        monitor = PerformanceMonitor()
        
        print("가짜 프레임 처리 시뮬레이션 (5초간)...")
        
        for i in range(20):
            start_time = time.time()
            
            # 가짜 처리 시간 (50-100ms)
            time.sleep(0.05 + (i % 5) * 0.01)
            
            processing_time = time.time() - start_time
            monitor.log_frame_time(processing_time)
            monitor.log_detection_time(processing_time * 0.8)  # 감지 시간은 80%
            
            if i % 5 == 0:
                print(f"  프레임 {i+1}/20 처리 완료")
        
        print("\n성능 통계:")
        monitor.print_stats()
        
        print("✓ 성능 모니터 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"✗ 성능 모니터 테스트 실패: {e}")
        return False

def test_usage_instructions():
    """사용법 안내 테스트"""
    print("사용법 안내 출력 테스트:")
    print_usage_instructions()
    return True

def run_all_util_tests():
    """모든 유틸리티 테스트 실행"""
    print("🧪 모든 유틸리티 테스트 실행")
    print("=" * 60)
    
    tests = [
        ("로깅 시스템", test_logging_system),
        ("시스템 체크", test_system_checks),
        ("모델 검증", test_model_validation),
        ("성능 모니터", test_performance_monitor),
        ("사용법 안내", test_usage_instructions)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name} 테스트 시작...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} 테스트 중 오류: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✓ 성공" if result else "✗ 실패"
        print(f"{test_name:15}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n총 {total_count}개 테스트 중 {success_count}개 성공")
    
    if success_count == total_count:
        print("🎉 모든 테스트 성공!")
        return True
    else:
        print("❌ 일부 테스트 실패!")
        return False

if __name__ == "__main__":
    import sys
    
    print("유틸리티 테스트 옵션:")
    print("1. 로깅 시스템 테스트")
    print("2. 시스템 체크 테스트")
    print("3. 모델 검증 테스트")
    print("4. 성능 모니터 테스트")
    print("5. 사용법 안내 테스트")
    print("6. 모든 테스트 실행")
    
    choice = input("선택 (1-6): ").strip()
    
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
        print("잘못된 선택입니다. 모든 테스트를 실행합니다.")
        success = run_all_util_tests()
    
    if success:
        print("\n🎉 테스트 성공!")
    else:
        print("\n❌ 테스트 실패!")
        sys.exit(1)