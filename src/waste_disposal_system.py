"""
라즈베리파이에서 확실히 작동하는 키보드 입력 처리
검증된 방법으로 교체
"""

import time
import threading
import logging
from typing import Optional, Dict, Any
from picamera2 import Picamera2
import numpy as np
import sys
import termios
import tty

from config import *  
from ppe_detector import PPEDetector
from servo_controller import ServoController, DoorState

class RaspberryPiKeyboardListener:
    """라즈베리파이에서 확실히 작동하는 키보드 리스너"""
    
    def __init__(self):
        self.old_settings = None
        self.running = False
        self.latest_char = None
        self.char_available = False
        self.input_thread = None
        self.lock = threading.Lock()
        
    def _getch(self):
        """단일 문자 읽기 (블로킹)"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    def _input_thread_function(self):
        """별도 스레드에서 키보드 입력 처리"""
        print("Keyboard listener thread started")
        while self.running:
            try:
                # 블로킹 방식으로 키 입력 대기
                char = self._getch()
                
                with self.lock:
                    self.latest_char = char
                    self.char_available = True
                    
                # 디버깅용 출력
                if char:
                    print(f"Key detected: '{char}' (ASCII: {ord(char)})")
                    
            except Exception as e:
                if self.running:
                    print(f"Keyboard input error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """키보드 리스너 시작"""
        if self.running:
            return
        
        print("Starting Raspberry Pi keyboard listener...")
        self.running = True
        self.input_thread = threading.Thread(target=self._input_thread_function, daemon=True)
        self.input_thread.start()
        print("Keyboard listener started - press keys to test")
    
    def stop(self):
        """키보드 리스너 중지"""
        print("Stopping keyboard listener...")
        self.running = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=2.0)
        print("Keyboard listener stopped")
    
    def get_char(self):
        """키보드 입력 가져오기 (논블로킹)"""
        with self.lock:
            if self.char_available:
                char = self.latest_char
                self.char_available = False
                self.latest_char = None
                return char
        return None
    
    def has_input(self):
        """입력이 있는지 확인"""
        with self.lock:
            return self.char_available

class WasteDisposalSystem:
    def __init__(self):
        """Initialize the waste disposal system"""
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.ppe_detector: Optional[PPEDetector] = None
        self.servo_controller: Optional[ServoController] = None
        self.camera: Optional[Picamera2] = None
        
        # Keyboard control (Raspberry Pi optimized)
        self.keyboard_listener = RaspberryPiKeyboardListener()
        self.detection_requested = False
        self.detection_in_progress = False
        
        # State tracking
        self.is_running = False
        self.compliance_start_time: Optional[float] = None
        self.door_open_time: Optional[float] = None
        self.last_detection_time = 0
        self.last_log_time = 0
        self.frame_count = 0
        self.fps = 0
        self.inference_active = False
        
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
            'start_time': None,
            'detection_sessions': 0,
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing waste disposal system with Raspberry Pi keyboard control...")
            
            # Initialize PPE detector (preload)
            self.logger.info("Loading PPE detection model...")
            self.ppe_detector = PPEDetector()
            self.logger.info("PPE detector loaded and ready!")
            
            # Initialize servo controller
            self.servo_controller = ServoController()
            self.logger.info("Servo controller initialized")
            
            # Initialize camera
            self._initialize_camera()
            self.logger.info("Camera initialized")
            
            # Start keyboard listener
            self.keyboard_listener.start()
            
            self.logger.info("System ready! Press SPACE to start PPE detection, Q to quit")
            print("\n" + "="*60)
            print("KEYBOARD CONTROLS:")
            print("  SPACE - Start PPE detection")
            print("  R     - Reset detection session")
            print("  S     - Show current status")
            print("  H     - Show help")
            print("  Q     - Quit system")
            print("="*60)
            
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
            time.sleep(2)
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            raise
    
    def _handle_keyboard_input(self):
        """Handle keyboard input"""
        char = self.keyboard_listener.get_char()
        if char:
            print(f"Processing key: '{char}'")  # 디버깅용
            
            if char == ' ':  # Space bar
                if not self.detection_in_progress:
                    self.detection_requested = True
                    self.stats['detection_sessions'] += 1
                    self.logger.info("PPE detection started!")
                    print(">>> PPE DETECTION STARTED <<<")
                else:
                    self.logger.info("Detection already in progress...")
                    print(">>> Detection already running <<<")
                    
            elif char.lower() == 'q':  # Q key to quit
                self.logger.info("Quit requested by user")
                print(">>> QUITTING SYSTEM <<<")
                self.stop_event.set()
                
            elif char.lower() == 'r':  # R key to reset
                self._reset_detection()
                self.logger.info("Detection session reset")
                print(">>> DETECTION RESET <<<")
                
            elif char.lower() == 'h':  # H key for help
                self._print_help()
                
            elif char.lower() == 's':  # S key for status
                self._print_status()
                
            else:
                print(f"Unknown key: '{char}' - Press H for help")
    
    def _reset_detection(self):
        """Reset detection session"""
        self.detection_requested = False
        self.detection_in_progress = False
        self.compliance_start_time = None
        if self.servo_controller and self.servo_controller.is_door_open():
            self.servo_controller.close_door()
            self.door_open_time = None
        self.logger.info("Detection session has been reset")
    
    def _print_help(self):
        """Print help message"""
        help_text = """
===========================================================
                    KEYBOARD CONTROLS
===========================================================
  SPACE  - Start PPE detection
  R      - Reset detection session  
  S      - Show current status
  H      - Show this help
  Q      - Quit system
===========================================================
        """
        print(help_text)
    
    def _print_status(self):
        """Print current status"""
        stats = self.get_statistics()
        status_text = f"""
===========================================================
                    CURRENT SYSTEM STATUS
===========================================================
  Runtime: {stats['runtime_seconds']:.1f} seconds
  Processed frames: {stats['total_frames']}
  Detection count: {stats['detection_count']}
  Door openings: {stats['door_openings']}
  Current FPS: {stats['current_fps']:.1f}
  Door state: {stats['door_state']}
  Detection active: {'Yes' if stats['detection_active'] else 'No'}
===========================================================
        """
        print(status_text)
    
    def _should_run_inference(self) -> tuple[bool, str]:
        """Determine if inference should run"""
        
        # 1. Door open - stop inference for servo stability
        if self.servo_controller.is_door_open():
            return False, "Door open - servo stability mode"
        
        # 2. Door moving - wait for completion
        if self.servo_controller.get_door_state() == DoorState.MOVING:
            return False, "Door moving - waiting for completion"
        
        # 3. User requested and not in progress
        if self.detection_requested and not self.detection_in_progress:
            self.detection_in_progress = True
            self.detection_requested = False  # Start only once
            return True, "User requested detection"
        
        # 4. Already in progress - continue
        if self.detection_in_progress:
            return True, "Detection session active"
        
        # 5. Default waiting state
        return False, "Waiting for user input (SPACE)"
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame with smart inference control"""
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        # Check if inference should run
        should_run, reason = self._should_run_inference()
        
        if not should_run:
            # Return empty result without inference
            return {
                'detections': [],
                'is_compliant': False,
                'ppe_status': {},
                'detection_summary': "Waiting for input",
                'inference_active': False,
                'status_reason': reason
            }
        
        # Fast PPE detection (model already loaded)
        start_time = time.time()
        detections = self.ppe_detector.detect(frame, CONFIDENCE_THRESHOLD)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        if detections:
            self.stats['detection_count'] += 1
        
        # Check PPE compliance
        is_compliant, ppe_status = self.ppe_detector.check_ppe_compliance(detections)
        
        return {
            'detections': detections,
            'is_compliant': is_compliant,
            'ppe_status': ppe_status,
            'detection_summary': self.ppe_detector.get_detection_summary(detections),
            'inference_active': True,
            'status_reason': reason,
            'inference_time_ms': inference_time
        }
    
    def _handle_compliance_state(self, result: Dict[str, Any], current_time: float):
        """Handle PPE compliance state changes"""
        # Only process when inference is active
        if not result.get('inference_active', False):
            return
        
        is_compliant = result['is_compliant']
        
        if is_compliant:
            if self.compliance_start_time is None:
                self.compliance_start_time = current_time
                self.logger.info("PPE compliance detected - starting timer")
                print(">>> PPE COMPLIANCE DETECTED <<<")
            
            compliance_duration = current_time - self.compliance_start_time
            
            # Open door when compliance achieved
            if (compliance_duration >= PPE_CHECK_DURATION and 
                self.servo_controller.is_door_closed()):
                
                self.logger.info(f"PPE compliance maintained for {compliance_duration:.1f}s - opening door")
                print(f">>> OPENING DOOR - PPE OK FOR {compliance_duration:.1f}s <<<")
                if self.servo_controller.open_door():
                    self.door_open_time = current_time
                    self.stats['door_openings'] += 1
                    self.stats['compliance_events'] += 1
                    self.detection_in_progress = False  # End detection session
                    self.logger.info("Door opened! Detection session completed.")
                    print(">>> DOOR OPENED - DETECTION COMPLETE <<<")
        else:
            if self.compliance_start_time is not None:
                self.logger.info("PPE compliance lost - resetting timer")
                print(">>> PPE COMPLIANCE LOST <<<")
                self.compliance_start_time = None
    
    def _handle_door_timeout(self, current_time: float):
        """Handle automatic door closing after timeout"""
        if (self.door_open_time is not None and 
            self.servo_controller.is_door_open()):
            
            door_open_duration = current_time - self.door_open_time
            
            # Warning messages
            if door_open_duration >= DOOR_OPEN_DURATION - 3 and door_open_duration < DOOR_OPEN_DURATION:
                remaining = DOOR_OPEN_DURATION - door_open_duration
                if int(remaining * 2) % 2 == 0:  # Every 0.5 seconds
                    self.logger.warning(f"WARNING: Door will close in {remaining:.1f} seconds")
                    print(f">>> WARNING: Door closes in {remaining:.1f}s <<<")
            
            # Auto close
            if door_open_duration >= DOOR_OPEN_DURATION:
                self.logger.info(f"Door timeout ({door_open_duration:.1f}s) - closing door")
                print(">>> AUTO-CLOSING DOOR <<<")
                if self.servo_controller.close_door():
                    self.door_open_time = None
                    self.compliance_start_time = None
                    self.detection_in_progress = False
                    self.logger.info("System ready for next detection (press SPACE)")
                    print(">>> READY FOR NEXT DETECTION (PRESS SPACE) <<<")
    
    def _update_fps(self, current_time: float):
        """Update FPS calculation"""
        if self.frame_count % FPS_UPDATE_INTERVAL == 0:
            time_diff = current_time - self.last_detection_time
            if time_diff > 0:
                self.fps = FPS_UPDATE_INTERVAL / time_diff
                
                if self.stats['start_time']:
                    total_time = current_time - self.stats['start_time']
                    self.stats['avg_fps'] = self.stats['total_frames'] / total_time
            
            self.last_detection_time = current_time
    
    def _log_status(self, result: Dict[str, Any], current_time: float):
        """Log current system status"""
        if not LOG_DETECTIONS and not DEBUG_MODE:
            return
        
        # More frequent logging when active
        log_interval = 1.0 if result.get('inference_active', False) else 3.0
        if current_time - self.last_log_time < log_interval:
            return
        
        self.last_log_time = current_time
        
        status_parts = []
        
        # FPS info
        status_parts.append(f"FPS: {self.fps:.1f}")
        
        # Door state
        door_state = self.servo_controller.get_door_state().value
        door_symbol = "[OPEN]" if door_state == "open" else "[CLOSED]"
        status_parts.append(f"Door: {door_symbol}")
        
        # Detection status
        if result.get('inference_active', False):
            inference_time = result.get('inference_time_ms', 0)
            status_parts.append(f"Detection: ACTIVE ({inference_time:.1f}ms)")
        else:
            status_parts.append(f"Detection: {result.get('status_reason', 'PAUSED')}")
        
        # Compliance status
        if result.get('inference_active', False):
            if result['is_compliant']:
                if self.compliance_start_time:
                    duration = current_time - self.compliance_start_time
                    progress = min(duration / PPE_CHECK_DURATION * 100, 100)
                    status_parts.append(f"Compliant: {duration:.1f}s/{PPE_CHECK_DURATION}s ({progress:.0f}%)")
                else:
                    status_parts.append("Compliant: Ready")
            else:
                status_parts.append("Non-compliant")
        
        # Detection summary
        if result['detections'] and result.get('inference_active', False):
            status_parts.append(f"Detected: {result['detection_summary']}")
        elif result.get('inference_active', False):
            status_parts.append("No detections")
        
        # Door timer
        if self.door_open_time:
            door_duration = current_time - self.door_open_time
            remaining = DOOR_OPEN_DURATION - door_duration
            status_parts.append(f"Auto-close: {remaining:.1f}s")
        
        self.logger.info(" | ".join(status_parts))
    
    def run(self):
        """Main system loop with keyboard control"""
        try:
            self.is_running = True
            self.stats['start_time'] = time.time()
            self.last_detection_time = time.time()
            self.last_log_time = time.time()
            
            self.logger.info("Waste disposal system started with Raspberry Pi keyboard control")
            self.logger.info(f"PPE check duration: {PPE_CHECK_DURATION}s")
            self.logger.info(f"Door open duration: {DOOR_OPEN_DURATION}s")
            self.logger.info("Controls: SPACE=Start detection, R=Reset, Q=Quit, H=Help")
            self.logger.info("System ready - press SPACE to start PPE detection")
            
            print("\n>>> SYSTEM READY - PRESS SPACE TO START PPE DETECTION <<<")
            
            while not self.stop_event.is_set():
                current_time = time.time()
                
                # Handle keyboard input
                self._handle_keyboard_input()
                
                # Always capture frame (lightweight)
                frame = self.camera.capture_array()
                
                # Smart processing (conditional inference)
                result = self._process_frame(frame)
                
                # Handle compliance state
                self._handle_compliance_state(result, current_time)
                
                # Handle door timeout
                self._handle_door_timeout(current_time)
                
                # Update FPS
                self._update_fps(current_time)
                
                # Log status
                self._log_status(result, current_time)
                
                # Smart delay
                if result.get('inference_active', False):
                    time.sleep(0.05)  # Fast when active
                else:
                    time.sleep(0.2)   # Slow when waiting
        
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
            print("\n>>> CTRL+C DETECTED - SHUTTING DOWN <<<")
        except Exception as e:
            self.logger.error(f"System error: {e}")
            print(f"\n>>> SYSTEM ERROR: {e} <<<")
            import traceback
            traceback.print_exc()
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
        print(">>> STOPPING SYSTEM <<<")
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
            
            # Stop keyboard listener
            self.keyboard_listener.stop()
            
            self.logger.info("System cleanup completed")
            print(">>> SYSTEM CLEANUP COMPLETED <<<")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            print(f">>> CLEANUP ERROR: {e} <<<")
    
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
            'compliance_duration': (current_time - self.compliance_start_time) if self.compliance_start_time else 0,
            'detection_active': self.detection_in_progress
        }
    
    def emergency_stop(self):
        """Emergency stop"""
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        print(">>> EMERGENCY STOP ACTIVATED <<<")
        
        if self.servo_controller:
            self.servo_controller.emergency_stop()
        
        self.stop()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ==========================================
# 간단한 키보드 테스트 (단독 실행용)
# ==========================================

def test_keyboard_only():
    """키보드 입력만 테스트"""
    print("=" * 50)
    print("RASPBERRY PI KEYBOARD TEST")
    print("=" * 50)
    print("Press keys to test. Press 'q' to quit.")
    print("-" * 50)
    
    listener = RaspberryPiKeyboardListener()
    listener.start()
    
    try:
        while True:
            char = listener.get_char()
            if char:
                print(f"Key pressed: '{char}' (ASCII: {ord(char)})")
                if char.lower() == 'q':
                    print("Quit requested")
                    break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nCtrl+C detected")
    finally:
        listener.stop()
        print("Keyboard test completed")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "keyboard-test":
        test_keyboard_only()
    else:
        # 기존 메인 실행
        print("Waste Disposal System - Raspberry Pi Edition")
        print("For keyboard test only: python waste_disposal_system.py keyboard-test")
        print("")
        
        try:
            system = WasteDisposalSystem()
            system.run()
        except Exception as e:
            print(f"System failed to start: {e}")
            import traceback
            traceback.print_exc()