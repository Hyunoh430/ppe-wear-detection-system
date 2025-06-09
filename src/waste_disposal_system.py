"""
깔끔한 출력을 위한 개선된 폐기물 처리 시스템
키보드 입력과 로그 출력이 겹치지 않도록 개선
"""

import time
import threading
import logging
import sys
import termios
import tty
from typing import Optional, Dict, Any
from picamera2 import Picamera2
import numpy as np

from config import *  
from ppe_detector import PPEDetector
from servo_controller import ServoController, DoorState

class CleanOutputManager:
    """깔끔한 출력 관리 클래스"""
    
    def __init__(self):
        self.output_lock = threading.Lock()
        self.last_status_line = ""
        self.status_update_time = 0
        
    def clear_line(self):
        """현재 줄 지우기"""
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
        
    def print_message(self, message, message_type="INFO"):
        """메시지 깔끔하게 출력"""
        with self.output_lock:
            self.clear_line()
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {message_type}: {message}")
            sys.stdout.flush()
    
    def print_status(self, status_text):
        """상태 메시지를 한 줄에 계속 업데이트"""
        with self.output_lock:
            current_time = time.time()
            # 1초에 한 번만 업데이트
            if current_time - self.status_update_time < 1.0:
                return
                
            self.clear_line()
            # 상태를 한 줄에 표시
            truncated_status = status_text[:75] + "..." if len(status_text) > 75 else status_text
            sys.stdout.write(f"\rSTATUS: {truncated_status}")
            sys.stdout.flush()
            self.last_status_line = truncated_status
            self.status_update_time = current_time
    
    def print_key_action(self, action):
        """키 액션을 강조해서 출력"""
        with self.output_lock:
            self.clear_line()
            print(f"\n>>> {action} <<<")
            sys.stdout.flush()

class RaspberryPiKeyboardListener:
    """라즈베리파이에서 확실히 작동하는 키보드 리스너"""
    
    def __init__(self, output_manager):
        self.old_settings = None
        self.running = False
        self.latest_char = None
        self.char_available = False
        self.input_thread = None
        self.lock = threading.Lock()
        self.output_manager = output_manager
        
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
        while self.running:
            try:
                # 블로킹 방식으로 키 입력 대기
                char = self._getch()
                
                with self.lock:
                    self.latest_char = char
                    self.char_available = True
                    
            except Exception as e:
                if self.running:
                    self.output_manager.print_message(f"Keyboard input error: {e}", "ERROR")
                time.sleep(0.1)
    
    def start(self):
        """키보드 리스너 시작"""
        if self.running:
            return
        
        self.output_manager.print_message("Starting Raspberry Pi keyboard listener...")
        self.running = True
        self.input_thread = threading.Thread(target=self._input_thread_function, daemon=True)
        self.input_thread.start()
        self.output_manager.print_message("Keyboard listener started")
    
    def stop(self):
        """키보드 리스너 중지"""
        self.output_manager.print_message("Stopping keyboard listener...")
        self.running = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=2.0)
        self.output_manager.print_message("Keyboard listener stopped")
    
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
        # 출력 관리자 먼저 생성
        self.output_manager = CleanOutputManager()
        
        # 로거 설정 (콘솔 출력 비활성화)
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.ppe_detector: Optional[PPEDetector] = None
        self.servo_controller: Optional[ServoController] = None
        self.camera: Optional[Picamera2] = None
        
        # Keyboard control (Raspberry Pi optimized)
        self.keyboard_listener = RaspberryPiKeyboardListener(self.output_manager)
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
            self.output_manager.print_message("Initializing waste disposal system...")
            
            # Initialize PPE detector (preload)
            self.output_manager.print_message("Loading PPE detection model...")
            self.ppe_detector = PPEDetector()
            self.output_manager.print_message("PPE detector loaded and ready!")
            
            # Initialize servo controller
            self.servo_controller = ServoController()
            self.output_manager.print_message("Servo controller initialized")
            
            # Initialize camera
            self._initialize_camera()
            self.output_manager.print_message("Camera initialized")
            
            # Start keyboard listener
            self.keyboard_listener.start()
            
            self.output_manager.print_message("System ready!")
            self._show_controls()
            
        except Exception as e:
            self.output_manager.print_message(f"System initialization failed: {e}", "ERROR")
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
            self.output_manager.print_message(f"Camera initialization failed: {e}", "ERROR")
            raise
    
    def _show_controls(self):
        """컨트롤 안내 출력"""
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS:")
        print("  SPACE - Start PPE detection")
        print("  R     - Reset detection session")
        print("  S     - Show current status")
        print("  H     - Show help")
        print("  Q     - Quit system")
        print("="*60)
        print("System ready - press SPACE to start PPE detection")
        print("Status updates will appear below:")
        print("-"*60)
    
    def _handle_keyboard_input(self):
        """Handle keyboard input"""
        char = self.keyboard_listener.get_char()
        if char:
            if char == ' ':  # Space bar
                if not self.detection_in_progress:
                    self.detection_requested = True
                    self.stats['detection_sessions'] += 1
                    self.output_manager.print_key_action("PPE DETECTION STARTED")
                else:
                    self.output_manager.print_key_action("Detection already running")
                    
            elif char.lower() == 'q':  # Q key to quit
                self.output_manager.print_key_action("QUITTING SYSTEM")
                self.stop_event.set()
                
            elif char.lower() == 'r':  # R key to reset
                self._reset_detection()
                self.output_manager.print_key_action("DETECTION RESET")
                
            elif char.lower() == 'h':  # H key for help
                self._show_controls()
                
            elif char.lower() == 's':  # S key for status
                self._print_detailed_status()
                
    def _reset_detection(self):
        """Reset detection session"""
        self.detection_requested = False
        self.detection_in_progress = False
        self.compliance_start_time = None
        if self.servo_controller and self.servo_controller.is_door_open():
            self.servo_controller.close_door()
            self.door_open_time = None
    
    def _print_detailed_status(self):
        """상세 상태 출력"""
        stats = self.get_statistics()
        print(f"\n" + "="*50)
        print("DETAILED SYSTEM STATUS")
        print("="*50)
        print(f"Runtime: {stats['runtime_seconds']:.1f} seconds")
        print(f"Processed frames: {stats['total_frames']}")
        print(f"Detection count: {stats['detection_count']}")
        print(f"Door openings: {stats['door_openings']}")
        print(f"Current FPS: {stats['current_fps']:.1f}")
        print(f"Door state: {stats['door_state']}")
        print(f"Detection active: {'Yes' if stats['detection_active'] else 'No'}")
        print("="*50)
    
    def _should_run_inference(self) -> tuple[bool, str]:
        """Determine if inference should run"""
        
        # 1. Door open - stop inference for servo stability
        if self.servo_controller.is_door_open():
            return False, "Door open"
        
        # 2. Door moving - wait for completion
        if self.servo_controller.get_door_state() == DoorState.MOVING:
            return False, "Door moving"
        
        # 3. User requested and not in progress
        if self.detection_requested and not self.detection_in_progress:
            self.detection_in_progress = True
            self.detection_requested = False  # Start only once
            return True, "Detection active"
        
        # 4. Already in progress - continue
        if self.detection_in_progress:
            return True, "Detection active"
        
        # 5. Default waiting state
        return False, "Waiting for SPACE"
    
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
                'detection_summary': "No detection",
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
                self.output_manager.print_key_action("PPE COMPLIANCE DETECTED")
            
            compliance_duration = current_time - self.compliance_start_time
            
            # Open door when compliance achieved
            if (compliance_duration >= PPE_CHECK_DURATION and 
                self.servo_controller.is_door_closed()):
                
                self.output_manager.print_key_action(f"OPENING DOOR - PPE OK FOR {compliance_duration:.1f}s")
                if self.servo_controller.open_door():
                    self.door_open_time = current_time
                    self.stats['door_openings'] += 1
                    self.stats['compliance_events'] += 1
                    self.detection_in_progress = False  # End detection session
                    self.output_manager.print_key_action("DOOR OPENED - DETECTION COMPLETE")
        else:
            if self.compliance_start_time is not None:
                self.output_manager.print_key_action("PPE COMPLIANCE LOST")
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
                    self.output_manager.print_key_action(f"WARNING: Door closes in {remaining:.1f}s")
            
            # Auto close
            if door_open_duration >= DOOR_OPEN_DURATION:
                self.output_manager.print_key_action("AUTO-CLOSING DOOR")
                if self.servo_controller.close_door():
                    self.door_open_time = None
                    self.compliance_start_time = None
                    self.detection_in_progress = False
                    self.output_manager.print_key_action("READY FOR NEXT DETECTION (PRESS SPACE)")
    
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
    
    def _update_status_display(self, result: Dict[str, Any], current_time: float):
        """상태를 한 줄로 깔끔하게 업데이트"""
        status_parts = []
        
        # FPS
        status_parts.append(f"FPS:{self.fps:.1f}")
        
        # Door state
        door_state = self.servo_controller.get_door_state().value
        door_symbol = "OPEN" if door_state == "open" else "CLOSED"
        status_parts.append(f"Door:{door_symbol}")
        
        # Detection status
        if result.get('inference_active', False):
            inference_time = result.get('inference_time_ms', 0)
            status_parts.append(f"Detection:ACTIVE({inference_time:.0f}ms)")
        else:
            status_parts.append(f"Detection:{result.get('status_reason', 'PAUSED')}")
        
        # Compliance status
        if result.get('inference_active', False):
            if result['is_compliant']:
                if self.compliance_start_time:
                    duration = current_time - self.compliance_start_time
                    progress = min(duration / PPE_CHECK_DURATION * 100, 100)
                    status_parts.append(f"PPE:OK({duration:.1f}s/{PPE_CHECK_DURATION}s)")
                else:
                    status_parts.append("PPE:Ready")
            else:
                status_parts.append("PPE:Missing")
        
        # Detection summary
        if result['detections'] and result.get('inference_active', False):
            status_parts.append(f"Found:{len(result['detections'])}")
        
        # Door timer
        if self.door_open_time:
            door_duration = current_time - self.door_open_time
            remaining = DOOR_OPEN_DURATION - door_duration
            status_parts.append(f"AutoClose:{remaining:.1f}s")
        
        status_text = " | ".join(status_parts)
        self.output_manager.print_status(status_text)
    
    def run(self):
        """Main system loop with keyboard control"""
        try:
            self.is_running = True
            self.stats['start_time'] = time.time()
            self.last_detection_time = time.time()
            self.last_log_time = time.time()
            
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
                
                # Update status display (깔끔한 한 줄 업데이트)
                self._update_status_display(result, current_time)
                
                # Smart delay
                if result.get('inference_active', False):
                    time.sleep(0.05)  # Fast when active
                else:
                    time.sleep(0.2)   # Slow when waiting
        
        except KeyboardInterrupt:
            self.output_manager.print_key_action("CTRL+C DETECTED - SHUTTING DOWN")
        except Exception as e:
            self.output_manager.print_message(f"System error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """Stop the system"""
        self.output_manager.print_key_action("STOPPING SYSTEM")
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
            
            self.output_manager.print_message("System cleanup completed")
            
        except Exception as e:
            self.output_manager.print_message(f"Cleanup error: {e}", "ERROR")
    
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
        self.output_manager.print_key_action("EMERGENCY STOP ACTIVATED")
        
        if self.servo_controller:
            self.servo_controller.emergency_stop()
        
        self.stop()

# ==========================================
# 키보드 테스트 (깔끔한 출력)
# ==========================================

def test_keyboard_only():
    """키보드 입력만 테스트 (깔끔한 출력)"""
    print("=" * 50)
    print("RASPBERRY PI KEYBOARD TEST - CLEAN OUTPUT")
    print("=" * 50)
    print("Press keys to test. Press 'q' to quit.")
    print("-" * 50)
    
    output_manager = CleanOutputManager()
    listener = RaspberryPiKeyboardListener(output_manager)
    listener.start()
    
    try:
        while True:
            char = listener.get_char()
            if char:
                output_manager.print_message(f"Key pressed: '{char}' (ASCII: {ord(char)})")
                if char.lower() == 'q':
                    output_manager.print_message("Quit requested")
                    break
            
            # 상태 업데이트 테스트
            output_manager.print_status(f"Waiting for input... Time: {time.strftime('%H:%M:%S')}")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        output_manager.print_message("Ctrl+C detected", "WARNING")
    finally:
        listener.stop()
        print("\nKeyboard test completed")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "keyboard-test":
        test_keyboard_only()
    else:
        try:
            system = WasteDisposalSystem()
            system.run()
        except Exception as e:
            print(f"System failed to start: {e}")
            import traceback
            traceback.print_exc()