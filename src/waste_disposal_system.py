"""
심플하고 깔끔한 폐기물 처리 시스템
복잡한 기능 제거, 간단한 출력만
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

class SimpleKeyboardListener:
    """간단한 키보드 리스너"""
    
    def __init__(self):
        self.running = False
        self.latest_char = None
        self.char_available = False
        self.input_thread = None
        self.lock = threading.Lock()
        
    def _getch(self):
        """단일 문자 읽기"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    def _input_thread_function(self):
        """키보드 입력 처리"""
        while self.running:
            try:
                char = self._getch()
                with self.lock:
                    self.latest_char = char
                    self.char_available = True
            except:
                time.sleep(0.1)
    
    def start(self):
        """시작"""
        self.running = True
        self.input_thread = threading.Thread(target=self._input_thread_function, daemon=True)
        self.input_thread.start()
    
    def stop(self):
        """중지"""
        self.running = False
        if self.input_thread:
            self.input_thread.join(timeout=1.0)
    
    def get_char(self):
        """키 입력 가져오기"""
        with self.lock:
            if self.char_available:
                char = self.latest_char
                self.char_available = False
                self.latest_char = None
                return char
        return None

class WasteDisposalSystem:
    def __init__(self):
        """시스템 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트들
        self.ppe_detector: Optional[PPEDetector] = None
        self.servo_controller: Optional[ServoController] = None
        self.camera: Optional[Picamera2] = None
        self.keyboard_listener = SimpleKeyboardListener()
        
        # 상태 변수들
        self.is_running = False
        self.detection_requested = False
        self.detection_in_progress = False
        self.compliance_start_time: Optional[float] = None
        self.door_open_time: Optional[float] = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = 0
        
        # 스레딩
        self.stop_event = threading.Event()
        
        # 통계
        self.stats = {
            'total_frames': 0,
            'detection_count': 0,
            'door_openings': 0,
            'start_time': None,
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """시스템 초기화"""
        try:
            print("Loading PPE detector...")
            self.ppe_detector = PPEDetector()
            
            print("Initializing servo...")
            self.servo_controller = ServoController()
            
            print("Starting camera...")
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": CAMERA_FORMAT}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(2)
            
            self.keyboard_listener.start()
            
            # 간단한 시작 메시지만
            print("\n" + "="*40)
            print("PPE WASTE DISPOSAL SYSTEM")
            print("="*40)
            print("Press SPACE to start PPE detection")
            print("Press Q to quit")
            print("="*40)
            
        except Exception as e:
            print(f"Initialization failed: {e}")
            raise
    
    def _handle_keyboard_input(self):
        """키보드 입력 처리"""
        char = self.keyboard_listener.get_char()
        if char:
            if char == ' ':  # SPACE
                if not self.detection_in_progress:
                    self.detection_requested = True
                    print(">>> PPE Detection Started <<<")
                else:
                    print(">>> Detection already running <<<")
                    
            elif char.lower() == 'q':  # Q
                print(">>> Quitting <<<")
                self.stop_event.set()
                
            elif char.lower() == 'r':  # R (리셋)
                self._reset_detection()
                print(">>> Reset <<<")
    
    def _reset_detection(self):
        """감지 리셋"""
        self.detection_requested = False
        self.detection_in_progress = False
        self.compliance_start_time = None
        if self.servo_controller and self.servo_controller.is_door_open():
            self.servo_controller.close_door()
            self.door_open_time = None
    
    def _should_run_inference(self) -> tuple[bool, str]:
        """추론 실행 여부"""
        if self.servo_controller.is_door_open():
            return False, "Door Open"
        
        if self.servo_controller.get_door_state() == DoorState.MOVING:
            return False, "Door Moving"
        
        if self.detection_requested and not self.detection_in_progress:
            self.detection_in_progress = True
            self.detection_requested = False
            return True, "Active"
        
        if self.detection_in_progress:
            return True, "Active"
        
        return False, "Wait for SPACE"
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """프레임 처리"""
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        should_run, reason = self._should_run_inference()
        
        if not should_run:
            return {
                'detections': [],
                'is_compliant': False,
                'inference_active': False,
                'status_reason': reason
            }
        
        # PPE 감지
        detections = self.ppe_detector.detect(frame, CONFIDENCE_THRESHOLD)
        if detections:
            self.stats['detection_count'] += 1
        
        # 준수 확인
        is_compliant, ppe_status = self.ppe_detector.check_ppe_compliance(detections)
        
        return {
            'detections': detections,
            'is_compliant': is_compliant,
            'inference_active': True,
            'status_reason': reason
        }
    
    def _handle_compliance_state(self, result: Dict[str, Any], current_time: float):
        """준수 상태 처리"""
        if not result.get('inference_active', False):
            return
        
        is_compliant = result['is_compliant']
        
        if is_compliant:
            if self.compliance_start_time is None:
                self.compliance_start_time = current_time
                print(">>> PPE Compliant - Timer Started <<<")
            
            duration = current_time - self.compliance_start_time
            
            if duration >= PPE_CHECK_DURATION and self.servo_controller.is_door_closed():
                print(f">>> Opening Door - PPE OK for {duration:.1f}s <<<")
                if self.servo_controller.open_door():
                    self.door_open_time = current_time
                    self.stats['door_openings'] += 1
                    self.detection_in_progress = False
                    print(">>> Door Opened <<<")
        else:
            if self.compliance_start_time is not None:
                print(">>> PPE Compliance Lost <<<")
                self.compliance_start_time = None
    
    def _handle_door_timeout(self, current_time: float):
        """문 타임아웃 처리"""
        if self.door_open_time and self.servo_controller.is_door_open():
            duration = current_time - self.door_open_time
            
            if duration >= DOOR_OPEN_DURATION - 2 and duration < DOOR_OPEN_DURATION:
                remaining = DOOR_OPEN_DURATION - duration
                if int(remaining * 2) % 2 == 0:
                    print(f">>> Door closes in {remaining:.1f}s <<<")
            
            if duration >= DOOR_OPEN_DURATION:
                print(">>> Auto-closing Door <<<")
                if self.servo_controller.close_door():
                    self.door_open_time = None
                    self.compliance_start_time = None
                    self.detection_in_progress = False
                    print(">>> Ready for next detection <<<")
    
    def _update_fps(self, current_time: float):
        """FPS 업데이트"""
        if current_time - self.last_fps_time >= 1.0:
            if self.last_fps_time > 0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
            self.last_fps_time = current_time
    
    def _print_status(self, result: Dict[str, Any]):
        """상태 출력 (한 줄로, 감지 내용 전체 표시)"""
        status_parts = []
        
        # FPS
        status_parts.append(f"FPS:{self.fps:.1f}")
        
        # 문 상태
        door_state = "OPEN" if self.servo_controller.is_door_open() else "CLOSED"
        status_parts.append(f"Door:{door_state}")
        
        # 감지 상태
        status_parts.append(f"Detection:{result.get('status_reason', 'Unknown')}")
        
        # 감지된 PPE (전체 표시, 줄임 없이)
        if result.get('inference_active', False) and result['detections']:
            detected_items = []
            for det in result['detections']:
                item_name = det['class_name'].replace('_', ' ')
                detected_items.append(f"{item_name}({det['confidence']:.2f})")
            # 모든 감지 항목 표시 (줄임표 없이)
            detection_text = ", ".join(detected_items)
            status_parts.append(f"Found: {detection_text}")
        elif result.get('inference_active', False):
            status_parts.append("Found: None")
        
        # PPE 준수 상태
        if result.get('inference_active', False):
            if result['is_compliant']:
                if self.compliance_start_time:
                    duration = time.time() - self.compliance_start_time
                    status_parts.append(f"PPE:OK({duration:.1f}s/{PPE_CHECK_DURATION}s)")
                else:
                    status_parts.append("PPE:Ready")
            else:
                status_parts.append("PPE:Missing")
        
        # 문 타이머
        if self.door_open_time:
            door_duration = time.time() - self.door_open_time
            remaining = DOOR_OPEN_DURATION - door_duration
            status_parts.append(f"AutoClose:{remaining:.1f}s")
        
        # 한 줄로 출력 (덮어쓰기)
        status_text = " | ".join(status_parts)
        sys.stdout.write(f'\r{status_text}' + ' ' * 20)
        sys.stdout.flush()
    
    def run(self):
        """메인 루프"""
        try:
            self.is_running = True
            self.stats['start_time'] = time.time()
            self.last_fps_time = time.time()
            
            while not self.stop_event.is_set():
                current_time = time.time()
                
                # 키보드 입력
                self._handle_keyboard_input()
                
                # 프레임 처리
                frame = self.camera.capture_array()
                result = self._process_frame(frame)
                
                # 준수 상태 처리
                self._handle_compliance_state(result, current_time)
                
                # 문 타임아웃
                self._handle_door_timeout(current_time)
                
                # FPS 업데이트
                self._update_fps(current_time)
                
                # 상태 출력
                self._print_status(result)
                
                # 딜레이
                if result.get('inference_active', False):
                    time.sleep(0.05)
                else:
                    time.sleep(0.2)
        
        except KeyboardInterrupt:
            print("\n>>> Ctrl+C - Shutting down <<<")
        except Exception as e:
            print(f"\nSystem error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """시스템 중지"""
        print("\n>>> Stopping System <<<")
        self.stop_event.set()
        self.is_running = False
        self.cleanup()
    
    def cleanup(self):
        """정리"""
        try:
            if self.servo_controller and self.servo_controller.is_door_open():
                self.servo_controller.close_door()
            
            if self.servo_controller:
                self.servo_controller.cleanup()
            
            if self.camera:
                self.camera.stop()
            
            self.keyboard_listener.stop()
            
            print("Cleanup completed")
            
        except Exception as e:
            print(f"Cleanup error: {e}")

if __name__ == "__main__":
    try:
        system = WasteDisposalSystem()
        system.run()
    except Exception as e:
        print(f"Failed to start: {e}")
        import traceback
        traceback.print_exc()