"""
Main waste disposal system integrating PPE detection and door control
"""

import time
import threading
import logging
from typing import Optional, Dict, Any
from picamera2 import Picamera2
import numpy as np

from .config import *
from .ppe_detector import PPEDetector
from .servo_controller import ServoController, DoorState

class WasteDisposalSystem:
    def __init__(self):
        """Initialize the waste disposal system"""
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.ppe_detector: Optional[PPEDetector] = None
        self.servo_controller: Optional[ServoController] = None
        self.camera: Optional[Picamera2] = None
        
        # State tracking
        self.is_running = False
        self.compliance_start_time: Optional[float] = None
        self.door_open_time: Optional[float] = None
        self.last_detection_time = 0
        self.frame_count = 0
        self.fps = 0
        
        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'detection_count': 0,
            'compliance_events': 0,
            'door_openings': 0,
            'avg_fps': 0,
            'start_time': None
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing waste disposal system...")
            
            # Initialize PPE detector
            self.ppe_detector = PPEDetector()
            self.logger.info("PPE detector initialized")
            
            # Initialize servo controller
            self.servo_controller = ServoController()
            self.logger.info("Servo controller initialized")
            
            # Initialize camera
            self._initialize_camera()
            self.logger.info("Camera initialized")
            
            # Test servo movement
            if not self.servo_controller.test_movement():
                raise Exception("Servo test failed")
            
            self.logger.info("Waste disposal system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.cleanup()
            raise
    
    def _initialize_camera(self):
        """Initialize Picamera2"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": CAMERA_FORMAT}
            )
            self.camera.configure(config)
            self.camera.start()
            
            # Wait for camera to stabilize
            time.sleep(2)
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            raise
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame for PPE detection"""
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        # Perform PPE detection
        detections = self.ppe_detector.detect(frame, CONFIDENCE_THRESHOLD)
        
        if detections:
            self.stats['detection_count'] += 1
        
        # Check PPE compliance
        is_compliant, ppe_status = self.ppe_detector.check_ppe_compliance(detections)
        
        # Create frame result
        result = {
            'detections': detections,
            'is_compliant': is_compliant,
            'ppe_status': ppe_status,
            'detection_summary': self.ppe_detector.get_detection_summary(detections)
        }
        
        return result
    
    def _handle_compliance_state(self, is_compliant: bool, current_time: float):
        """Handle PPE compliance state changes"""
        if is_compliant:
            # Start or continue compliance timing
            if self.compliance_start_time is None:
                self.compliance_start_time = current_time
                self.logger.info("PPE compliance detected - starting timer")
            
            compliance_duration = current_time - self.compliance_start_time
            
            # Check if compliance duration threshold is met
            if (compliance_duration >= PPE_CHECK_DURATION and 
                self.servo_controller.is_door_closed()):
                
                self.logger.info(f"PPE compliance maintained for {compliance_duration:.1f}s - opening door")
                if self.servo_controller.open_door():
                    self.door_open_time = current_time
                    self.stats['door_openings'] += 1
                    self.stats['compliance_events'] += 1
        else:
            # Reset compliance timing
            if self.compliance_start_time is not None:
                self.logger.info("PPE compliance lost - resetting timer")
                self.compliance_start_time = None
    
    def _handle_door_timeout(self, current_time: float):
        """Handle automatic door closing after timeout"""
        if (self.door_open_time is not None and 
            self.servo_controller.is_door_open()):
            
            door_open_duration = current_time - self.door_open_time
            
            if door_open_duration >= DOOR_OPEN_DURATION:
                self.logger.info(f"Door open timeout ({door_open_duration:.1f}s) - closing door")
                if self.servo_controller.close_door():
                    self.door_open_time = None
    
    def _update_fps(self, current_time: float):
        """Update FPS calculation"""
        if self.frame_count % FPS_UPDATE_INTERVAL == 0:
            time_diff = current_time - self.last_detection_time
            if time_diff > 0:
                self.fps = FPS_UPDATE_INTERVAL / time_diff
                
                # Update average FPS
                if self.stats['start_time']:
                    total_time = current_time - self.stats['start_time']
                    self.stats['avg_fps'] = self.stats['total_frames'] / total_time
            
            self.last_detection_time = current_time
    
    def _log_status(self, result: Dict[str, Any], current_time: float):
        """Log current system status"""
        if not LOG_DETECTIONS and not DEBUG_MODE:
            return
        
        status_parts = []
        
        # FPS info
        status_parts.append(f"FPS: {self.fps:.1f}")
        
        # Door state
        door_state = self.servo_controller.get_door_state().value
        status_parts.append(f"Door: {door_state}")
        
        # Compliance status
        if result['is_compliant']:
            if self.compliance_start_time:
                duration = current_time - self.compliance_start_time
                status_parts.append(f"Compliant: {duration:.1f}s/{PPE_CHECK_DURATION}s")
            else:
                status_parts.append("Compliant: Starting")
        else:
            status_parts.append("Non-compliant")
        
        # Detection summary
        if result['detections']:
            status_parts.append(f"Detected: {result['detection_summary']}")
        else:
            status_parts.append("No detections")
        
        # Door open timer
        if self.door_open_time:
            door_duration = current_time - self.door_open_time
            status_parts.append(f"Door open: {door_duration:.1f}s/{DOOR_OPEN_DURATION}s")
        
        self.logger.info(" | ".join(status_parts))
    
    def run(self):
        """Main system loop"""
        try:
            self.is_running = True
            self.stats['start_time'] = time.time()
            self.last_detection_time = time.time()
            
            self.logger.info("Starting waste disposal system main loop")
            self.logger.info(f"PPE check duration: {PPE_CHECK_DURATION}s")
            self.logger.info(f"Door open duration: {DOOR_OPEN_DURATION}s")
            self.logger.info("Press Ctrl+C to stop")
            
            while not self.stop_event.is_set():
                current_time = time.time()
                
                # Capture frame
                frame = self.camera.capture_array()
                
                # Process frame
                result = self._process_frame(frame)
                
                # Handle compliance state
                self._handle_compliance_state(result['is_compliant'], current_time)
                
                # Handle door timeout
                self._handle_door_timeout(current_time)
                
                # Update FPS
                self._update_fps(current_time)
                
                # Log status
                if self.frame_count % FPS_UPDATE_INTERVAL == 0:
                    self._log_status(result, current_time)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self.stop()
    
    def run_async(self):
        """Run system in separate thread"""
        if self.is_running:
            self.logger.warning("System already running")
            return
        
        self.main_thread = threading.Thread(target=self.run, daemon=True)
        self.main_thread.start()
        self.logger.info("System started in background thread")
    
    def stop(self):
        """Stop the system"""
        self.logger.info("Stopping waste disposal system...")
        self.stop_event.set()
        self.is_running = False
        
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up system resources"""
        try:
            # Close door if open
            if self.servo_controller and self.servo_controller.is_door_open():
                self.servo_controller.close_door()
            
            # Clean up servo
            if self.servo_controller:
                self.servo_controller.cleanup()
            
            # Stop camera
            if self.camera:
                self.camera.stop()
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        current_time = time.time()
        runtime = current_time - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            **self.stats,
            'runtime_seconds': runtime,
            'current_fps': self.fps,
            'door_state': self.servo_controller.get_door_state().value if self.servo_controller else 'unknown',
            'is_compliant': self.compliance_start_time is not None,
            'compliance_duration': (current_time - self.compliance_start_time) if self.compliance_start_time else 0
        }
    
    def emergency_stop(self):
        """Emergency stop - immediately stop all operations"""
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        
        if self.servo_controller:
            self.servo_controller.emergency_stop()
        
        self.stop()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


# ==========================================
# 개별 테스트 코드
# ==========================================

def test_system_initialization():
    """시스템 초기화 테스트"""
    print("=" * 50)
    print("폐기물 처리 시스템 초기화 테스트")
    print("=" * 50)
    
    try:
        print("1. 시스템 컴포넌트 초기화...")
        system = WasteDisposalSystem()
        
        print("   ✓ PPE 감지기 초기화 완료")
        print("   ✓ 서보 컨트롤러 초기화 완료")
        print("   ✓ 카메라 초기화 완료")
        
        print("\n2. 시스템 상태 확인...")
        stats = system.get_statistics()
        print(f"   도어 상태: {stats['door_state']}")
        print(f"   컴플라이언스 상태: {stats['is_compliant']}")
        
        print("\n3. 시스템 정리...")
        system.cleanup()
        print("   ✓ 리소스 정리 완료")
        
        print("\n✓ 시스템 초기화 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"✗ 시스템 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_short_run():
    """시스템 단기 실행 테스트 (30초)"""
    print("=" * 50)
    print("시스템 단기 실행 테스트 (30초)")
    print("=" * 50)
    print("PPE를 착용하고 테스트해보세요!")
    print("Ctrl+C로 중간에 중단할 수 있습니다.")
    
    try:
        import time
        import threading
        
        system = WasteDisposalSystem()
        
        # 백그라운드에서 시스템 실행
        system.run_async()
        
        # 30초 동안 상태 모니터링
        start_time = time.time()
        
        while time.time() - start_time < 30:
            time.sleep(5)  # 5초마다 상태 출력
            
            stats = system.get_statistics()
            elapsed = time.time() - start_time
            
            print(f"\n[{elapsed:.0f}s] 시스템 상태:")
            print(f"  - 처리 프레임: {stats['total_frames']}")
            print(f"  - 감지 횟수: {stats['detection_count']}")
            print(f"  - 컴플라이언스 이벤트: {stats['compliance_events']}")
            print(f"  - 도어 열림 횟수: {stats['door_openings']}")
            print(f"  - 현재 FPS: {stats['current_fps']:.1f}")
            print(f"  - 도어 상태: {stats['door_state']}")
        
        print("\n30초 테스트 완료!")
        system.stop()
        
        # 최종 통계
        final_stats = system.get_statistics()
        print(f"\n📊 최종 통계:")
        print(f"  - 총 실행 시간: {final_stats['runtime_seconds']:.1f}초")
        print(f"  - 총 프레임: {final_stats['total_frames']}")
        print(f"  - 평균 FPS: {final_stats['avg_fps']:.1f}")
        print(f"  - 감지 횟수: {final_stats['detection_count']}")
        print(f"  - 도어 열림: {final_stats['door_openings']}회")
        
        print("\n✓ 단기 실행 테스트 성공!")
        return True
        
    except KeyboardInterrupt:
        print("\n사용자가 테스트를 중단했습니다.")
        system.stop()
        return True
    except Exception as e:
        print(f"✗ 시스템 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        try:
            system.stop()
        except:
            pass
        return False

def test_system_components_integration():
    """시스템 컴포넌트 통합 테스트"""
    print("=" * 50)
    print("시스템 컴포넌트 통합 테스트")
    print("=" * 50)
    
    try:
        system = WasteDisposalSystem()
        
        print("1. PPE 감지기 단독 테스트...")
        frame = system.camera.capture_array()
        detections = system.ppe_detector.detect(frame)
        print(f"   감지된 객체 수: {len(detections)}")
        
        if detections:
            for det in detections:
                print(f"   - {det['class_name']}: {det['confidence']:.2f}")
        
        print("\n2. 서보 컨트롤러 단독 테스트...")
        print("   도어 열기...")
        if system.servo_controller.open_door():
            print("   ✓ 도어 열기 성공")
        
        import time
        time.sleep(2)
        
        print("   도어 닫기...")
        if system.servo_controller.close_door():
            print("   ✓ 도어 닫기 성공")
        
        print("\n3. PPE 컴플라이언스 체크 테스트...")
        is_compliant, ppe_status = system.ppe_detector.check_ppe_compliance(detections)
        print(f"   현재 컴플라이언스: {'✓' if is_compliant else '✗'}")
        print(f"   PPE 상태: {ppe_status}")
        
        print("\n4. 시스템 정리...")
        system.cleanup()
        
        print("\n✓ 컴포넌트 통합 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"✗ 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emergency_stop():
    """비상 정지 기능 테스트"""
    print("=" * 50)
    print("비상 정지 기능 테스트")
    print("=" * 50)
    
    try:
        system = WasteDisposalSystem()
        
        print("1. 시스템 시작...")
        system.run_async()
        
        import time
        time.sleep(3)
        
        print("2. 비상 정지 테스트...")
        system.emergency_stop()
        
        print("3. 시스템 상태 확인...")
        door_state = system.servo_controller.get_door_state()
        print(f"   도어 상태: {door_state.value}")
        
        if door_state.value == "error":
            print("   ✓ 비상 정지 성공 (ERROR 상태)")
        else:
            print("   ⚠ 비상 정지 상태 확인 필요")
        
        print("\n✓ 비상 정지 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"✗ 비상 정지 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    print("폐기물 처리 시스템 테스트 옵션:")
    print("1. 시스템 초기화 테스트")
    print("2. 단기 실행 테스트 (30초)")
    print("3. 컴포넌트 통합 테스트")
    print("4. 비상 정지 테스트")
    
    choice = input("선택 (1-4): ").strip()
    
    if choice == "1":
        success = test_system_initialization()
    elif choice == "2":
        success = test_system_short_run()
    elif choice == "3":
        success = test_system_components_integration()
    elif choice == "4":
        success = test_emergency_stop()
    else:
        print("잘못된 선택입니다. 초기화 테스트를 실행합니다.")
        success = test_system_initialization()
    
    if success:
        print("\n🎉 테스트 성공!")
    else:
        print("\n❌ 테스트 실패!")
        sys.exit(1)