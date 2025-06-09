"""
로그 메시지가 화면을 밀어내는 문제 해결
초기 메시지만 표시하고 이후에는 상태 바만 업데이트
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

class FixedOutputManager:
    """로그 밀림 문제를 해결한 출력 관리 클래스"""
    
    def __init__(self):
        self.output_lock = threading.Lock()
        self.initialization_complete = False
        self.status_update_time = 0
        
    def clear_line(self):
        """현재 줄 지우기"""
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        sys.stdout.flush()
        
    def print_init_message(self, message, message_type="INFO"):
        """초기화 중에만 메시지 출력"""
        if not self.initialization_complete:
            with self.output_lock:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {message_type}: {message}")
                sys.stdout.flush()
    
    def mark_initialization_complete(self):
        """초기화 완료 표시 - 이후 메시지 출력 중단"""
        self.initialization_complete = True
        
    def print_status(self, status_text):
        """상태 메시지를 한 줄에 계속 업데이트 (덮어쓰기)"""
        if not self.initialization_complete:
            return
            
        with self.output_lock:
            current_time = time.time()
            # 0.5초에 한 번만 업데이트
            if current_time - self.status_update_time < 0.5:
                return
                
            self.clear_line()
            # 상태를 한 줄에 표시 (길이 제한)
            truncated_status = status_text[:90] + "..." if len(status_text) > 90 else status_text
            sys.stdout.write(f"\rSTATUS: {truncated_status}")
            sys.stdout.flush()
            self.status_update_time = current_time
    
    def print_key_action(self, action):
        """키 액션만 새 줄에 출력"""
        with self.output_lock:
            self.clear_line()
            print(f"\n>>> {action} <<<")
            sys.stdout.flush()

class QuietKeyboardListener:
    """조용한 키보드 리스너 (초기화 메시지 최소화)"""
    
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
        """별도 스레드에서 키보드 입력 처리 (에러만 출력)"""
        while self.running:
            try:
                char = self._getch()
                with self.lock:
                    self.latest_char = char
                    self.char_available = True
            except Exception as e:
                if self.running:
                    # 초기화 완료 후에는 에러도 출력 안함
                    if not self.output_manager.initialization_complete:
                        self.output_manager.print_init_message(f"Keyboard error: {e}", "ERROR")
                time.sleep(0.1)
    
    def start(self):
        """키보드 리스너 시작 (메시지 최소화)"""
        if self.running:
            return
        
        self.output_manager.print_init_message("Starting keyboard listener...")
        self.running = True
        self.input_thread = threading.Thread(target=self._input_thread_function, daemon=True)
        self.input_thread.start()
        self.output_manager.print_init_message("Keyboard listener ready")
    
    def stop(self):
        """키보드 리스너 중지 (메시지 없음)"""
        self.running = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
    
    def get_char(self):
        """키보드 입력 가져오기 (논블로킹)"""
        with self.lock:
            if self.char_available:
                char = self.latest_char
                self.char_available = False
                self.latest_char = None
                return char
        return None

class WasteDisposalSystem:
    def __init__(self):
        """Initialize the waste disposal system"""
        # 출력 관리자 먼저 생성
        self.output_manager = FixedOutputManager()
        
        # 로거 설정 - 파일로만 출력하고 콘솔은 끄기
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.ppe_detector: Optional[PPEDetector] = None
        self.servo_controller: Optional[ServoController] = None
        self.camera: Optional[Picamera2] = None
        
        # Keyboard control
        self.keyboard_listener = QuietKeyboardListener(self.output_manager)
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
            self.output_manager.print_init_message("Initializing waste disposal system...")
            
            # Initialize PPE detector (preload)
            self.output_manager.print_init_message("Loading PPE detection model...")
            self.ppe_detector = PPEDetector()
            self.output_manager.print_init_message("PPE detector loaded")
            
            # Initialize servo controller
            self.servo_controller = ServoController()
            self.output_manager.print_init_message("Servo controller initialized")
            
            # Initialize camera
            self._initialize_camera()
            self.output_manager.print_init_message("Camera initialized")
            
            # Start keyboard listener
            self.keyboard_listener.start()
            
            self.output_manager.print_init_message("System ready!")
            
            # 초기화 완료 - 이제 화면 고정
            self.output_manager.mark_initialization_complete()
            self._show_final_interface()
            
        except Exception as e:
            self.output_manager.print_init_message(f"System initialization failed: {e}", "ERROR")
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
            self.output_manager.print_init_message(f"Camera initialization failed: {e}", "ERROR")
            raise
    
    def _show_final_interface(self):
        """최종 인터페이스 표시 (고정)"""
        print("\n" + "="*70)
        print("WASTE DISPOSAL SYSTEM - READY")
        print("="*70)
        print("KEYBOARD CONTROLS:")
        print("  SPACE - Start PPE detection")
        print("  R     - Reset detection session")
        print("  S     - Show detailed status")
        print("  H     - Show help")
        print("  Q     - Quit system")
        print("="*70)
        print("System is ready. Press SPACE to start PPE detection.")
        print("Status will be shown below:")
        print("-"*70)
    
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
                self._show_help_overlay()
                
            elif char.lower() == 's':  # S key for status
                self._show_detailed_status()
                
    def _reset_detection(self):
        """Reset detection session"""
        self.detection_requested = False
        self.detection_in_progress = False
        self.compliance_start_time = None
        if self.servo_controller and self.servo_controller.is_door_open():
            self.servo_controller.close_door()
            self.door_open_time = None
    
    def _show_help_overlay(self):
        """도움말 오버레이 (기존 화면 유지)"""
        print(f"\n" + "="*50)
        print("HELP - KEYBOARD CONTROLS")
        print("="*50)
        print("SPACE - Start PPE detection")
        print("R     - Reset detection session")
        print("S     - Show detailed status")
        print("H     - Show this help")
        print("Q     - Quit system")
        print("="*50)
    
    def _show_detailed_status(self):
        """상세 상태 출력 (오버레이 형식)"""
        stats = self.get_statistics()
        print(f"\n" + "="*60)
        print("DETAILED SYSTEM STATUS")
        print("="*60)
        print(f"Runtime: {stats['runtime_seconds']:.1f} seconds")
        print(f"Processed frames: {stats['total_frames']}")
        print(f"Detection count: {stats['detection_count']}")
        print(f"Door openings: {stats['door_openings']}")
        print(f"Current FPS: {stats['current_fps']:.1f}")
        print(f"Door state: {stats['door_state']}")
        print(f"Detection active: {'Yes' if stats['detection_active'] else 'No'}")
        print(f"Detection sessions: {stats['detection_sessions']}")
        print("="*60)
    
    def _should_run_inference(self) -> tuple[bool, str]:
        """Determine if inference should run"""
        
        if self.servo_controller.is_door_open():
            return False, "Door open"
        
        if self.servo_controller.get_door_state() == DoorState.MOVING:
            return False, "Door moving"
        
        if self.detection_requested and not self.detection_in_progress:
            self.detection_in_progress = True
            self.detection_requested = False
            return True, "Detection active"
        
        if self.detection_in_progress:
            return True, "Detection active"
        
        return False, "Waiting for SPACE"
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame with smart inference control"""
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        should_run, reason = self._should_run_inference()
        
        if not should_run:
            return {
                'detections': [],
                'is_compliant': False,
                'ppe_status': {},
                'detection_summary': "No detection",
                'inference_active': False,
                'status_reason': reason
            }
        
        start_time = time.time()
        detections = self.ppe_detector.detect(frame, CONFIDENCE_THRESHOLD)
        inference_time = (time.time() - start_time) * 1000
        
        if detections:
            self.stats['detection_count'] += 1
        
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
        if not result.get('inference_active', False):
            return
        
        is_compliant = result['is_compliant']
        
        if is_compliant:
            if self.compliance_start_time is None:
                self.compliance_start_time = current_time
                self.output_manager.print_key_action("PPE COMPLIANCE DETECTED")
            
            compliance_duration = current_time - self.compliance_start_time
            
            if (compliance_duration >= PPE_CHECK_DURATION and 
                self.servo_controller.is_door_closed()):
                
                self.output_manager.print_key_action(f"OPENING DOOR - PPE OK FOR {compliance_duration:.1f}s")
                if self.servo_controller.open_door():
                    self.door_open_time = current_time
                    self.stats['door_openings'] += 1
                    self.stats['compliance_events'] += 1
                    self.detection_in_progress = False
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
            
            if door_open_duration >= DOOR_OPEN_DURATION - 3 and door_open_duration < DOOR_OPEN_DURATION:
                remaining = DOOR_OPEN_DURATION - door_open_duration
                if int(remaining * 2) % 2 == 0:
                    self.output_manager.print_key_action(f"WARNING: Door closes in {remaining:.1f}s")
            
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
                
                self._handle_keyboard_input()
                frame = self.camera.capture_array()
                result = self._process_frame(frame)
                self._handle_compliance_state(result, current_time)
                self._handle_door_timeout(current_time)
                self._update_fps(current_time)
                self._update_status_display(result, current_time)
                
                if result.get('inference_active', False):
                    time.sleep(0.05)
                else:
                    time.sleep(0.2)
        
        except KeyboardInterrupt:
            self.output_manager.print_key_action("CTRL+C DETECTED - SHUTTING DOWN")
        except Exception as e:
            print(f"\nSYSTEM ERROR: {e}")
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
            if self.servo_controller and self.servo_controller.is_door_open():
                self.servo_controller.close_door()
            
            if self.servo_controller:
                self.servo_controller.cleanup()
            
            if self.camera:
                self.camera.stop()
            
            self.keyboard_listener.stop()
            
            print("\nSystem cleanup completed")
            
        except Exception as e:
            print(f"Cleanup error: {e}")
    
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

if __name__ == "__main__":
    try:
        system = WasteDisposalSystem()
        system.run()
    except Exception as e:
        print(f"System failed to start: {e}")
        import traceback
        traceback.print_exc()