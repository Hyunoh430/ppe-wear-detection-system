"""
PPE 감지 + 영상 전송 + 화재 알림 통합 시스템
- 유휴 상태: 영상 전송
- PPE 감지 중: 영상 전송 중단
- 화재 알림: UDP로 수신하여 음성 경고
"""

import time
import threading
import logging
import sys
import termios
import tty
import os
import socket
import struct
import cv2
from typing import Optional, Dict, Any
from picamera2 import Picamera2
import numpy as np

from config import *  
from ppe_detector import PPEDetector
from servo_controller import ServoController, DoorState

# 모든 로깅 완전 차단
logging.getLogger().setLevel(logging.CRITICAL + 1)
logging.getLogger('picamera2').setLevel(logging.CRITICAL + 1)
logging.getLogger('libcamera').setLevel(logging.CRITICAL + 1)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL + 1)
logging.getLogger('tflite_runtime').setLevel(logging.CRITICAL + 1)
logging.getLogger('RPi').setLevel(logging.CRITICAL + 1)
logging.getLogger('GPIO').setLevel(logging.CRITICAL + 1)

# stdout 백업 및 null 디바이스 준비
original_stdout = sys.stdout
original_stderr = sys.stderr

class NullOutput:
    """아무것도 출력하지 않는 클래스"""
    def write(self, txt): pass
    def flush(self): pass

def silence_all_output():
    """모든 출력 차단"""
    sys.stdout = NullOutput()
    sys.stderr = NullOutput()

def restore_output():
    """출력 복원"""
    sys.stdout = original_stdout
    sys.stderr = original_stderr

class FireAlertListener:
    """화재 알림 UDP 리스너"""
    
    def __init__(self, port: int = 8888):
        self.port = port
        self.running = False
        self.thread = None
        self.socket = None
        self.last_alert_time = 0
        self.alert_cooldown = 10  # 10초 쿨다운
        
    def start(self):
        """리스너 시작"""
        self.running = True
        self.thread = threading.Thread(target=self._listen_for_alerts, daemon=True)
        self.thread.start()
        
    def stop(self):
        """리스너 중지"""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _listen_for_alerts(self):
        """화재 알림 수신 스레드"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(1.0)  # 1초 타임아웃
            
            while self.running:
                try:
                    data, addr = self.socket.recvfrom(1024)
                    current_time = time.time()
                    
                    # 쿨다운 체크 (중복 알림 방지)
                    if current_time - self.last_alert_time < self.alert_cooldown:
                        continue
                    
                    if data == b"FIRE_DETECTED":
                        self._handle_fire_alert(addr[0])
                        self.last_alert_time = current_time
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:  # 정상 종료가 아닌 경우만 출력
                        print(f"Fire alert listener error: {e}")
                    
        except Exception as e:
            print(f"Failed to start fire alert listener: {e}")
        finally:
            if self.socket:
                self.socket.close()
    
    def _handle_fire_alert(self, sender_ip: str):
        """화재 알림 처리"""
        try:
            # 백그라운드에서 음성 경고 실행
            os.system("espeak 'Fire detected! Emergency evacuation required!' -s 150 &")
            print(f"\n🔥 FIRE ALERT from {sender_ip} - Voice warning activated!")
        except Exception as e:
            print(f"Error playing fire alert: {e}")

class QuietKeyboardListener:
    """완전히 조용한 키보드 리스너"""
    
    def __init__(self):
        self.running = False
        self.latest_char = None
        self.char_available = False
        self.input_thread = None
        self.lock = threading.Lock()
        
    def _getch(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    def _input_thread_function(self):
        while self.running:
            try:
                char = self._getch()
                with self.lock:
                    self.latest_char = char
                    self.char_available = True
            except:
                time.sleep(0.1)
    
    def start(self):
        self.running = True
        self.input_thread = threading.Thread(target=self._input_thread_function, daemon=True)
        self.input_thread.start()
    
    def stop(self):
        self.running = False
        if self.input_thread:
            self.input_thread.join(timeout=0.5)
    
    def get_char(self):
        with self.lock:
            if self.char_available:
                char = self.latest_char
                self.char_available = False
                self.latest_char = None
                return char
        return None

class VideoStreamer:
    """영상 전송 클래스"""
    
    def __init__(self, target_ip: str = "172.20.10.4", port: int = 9999):
        self.target_ip = target_ip
        self.port = port
        self.socket = None
        self.connected = False
        self.streaming = False
        
    def connect(self):
        """컴퓨터에 연결"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.target_ip, self.port))
            self.connected = True
            return True
        except Exception as e:
            self.connected = False
            return False
    
    def disconnect(self):
        """연결 해제"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
        self.streaming = False
    
    def send_frame(self, frame: np.ndarray, quiet_mode: bool = False):
        """프레임 전송"""
        if not self.connected:
            return False
        
        try:
            # 추론 중일 때는 모든 출력 차단
            if quiet_mode:
                silence_all_output()
            
            # JPEG 압축
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            data = buffer.tobytes()
            
            # 데이터 크기 먼저 전송
            size = len(data)
            self.socket.sendall(struct.pack('!I', size))
            
            # 실제 프레임 데이터 전송
            self.socket.sendall(data)
            
            if quiet_mode:
                restore_output()
            
            return True
        except Exception as e:
            if quiet_mode:
                restore_output()
            self.disconnect()
            return False
    
    def start_streaming(self):
        """스트리밍 시작"""
        self.streaming = True
    
    def stop_streaming(self):
        """스트리밍 중지"""
        self.streaming = False

class WasteDisposalSystem:
    def __init__(self, enable_video_streaming: bool = True, target_computer_ip: str = "172.20.10.4"):
        # 로거 완전 비활성화
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)
        
        # 컴포넌트들
        self.ppe_detector: Optional[PPEDetector] = None
        self.servo_controller: Optional[ServoController] = None
        self.camera: Optional[Picamera2] = None
        self.keyboard_listener = QuietKeyboardListener()
        
        # 화재 알림 리스너 추가
        self.fire_alert_listener = FireAlertListener()
        
        # 영상 스트리밍
        self.enable_video_streaming = enable_video_streaming
        self.video_streamer: Optional[VideoStreamer] = None
        if enable_video_streaming:
            self.video_streamer = VideoStreamer(target_computer_ip)
        
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
            'frames_streamed': 0,
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """시스템 초기화 - 모든 출력 차단하고 초기화"""
        try:
            # 모든 출력 차단
            silence_all_output()
            
            # 조용히 초기화
            self.ppe_detector = PPEDetector()
            self.servo_controller = ServoController()
            
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": CAMERA_FORMAT}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(2)
            
            self.keyboard_listener.start()
            
            # 화재 알림 리스너 시작
            self.fire_alert_listener.start()
            
            # 영상 스트리밍 연결 시도
            if self.video_streamer:
                if self.video_streamer.connect():
                    self.video_streamer.start_streaming()
            
            # 출력 복원 (상태바만 나오게)
            restore_output()
            
        except Exception as e:
            restore_output()
            print(f"Error: {e}")
            raise
    
    def _handle_keyboard_input(self):
        """키보드 입력 처리"""
        char = self.keyboard_listener.get_char()
        if char:
            if char == ' ':  # SPACE
                if not self.detection_in_progress:
                    self.detection_requested = True
                    
            elif char.lower() == 'q':  # Q
                self.stop_event.set()
                
            elif char.lower() == 'r':  # R (리셋)
                self._reset_detection()
    
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
            return False, "Moving"
        
        if self.detection_requested and not self.detection_in_progress:
            self.detection_in_progress = True
            self.detection_requested = False
            return True, "Active"
        
        if self.detection_in_progress:
            return True, "Active"
        
        return False, "Streaming"  # 유휴 상태일 때는 스트리밍 중
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """프레임 처리 - PPE 감지 또는 영상 전송"""
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        should_run, reason = self._should_run_inference()
        
        # 영상 스트리밍 제어
        if self.video_streamer:
            if should_run:
                # PPE 감지 중: 스트리밍 중지
                if self.video_streamer.streaming:
                    self.video_streamer.stop_streaming()
            else:
                # 유휴 상태: 스트리밍 시작/유지
                if not self.video_streamer.streaming:
                    self.video_streamer.start_streaming()
                
                # 프레임 전송 (일반 모드 - 로그 출력 허용)
                if self.video_streamer.streaming:
                    if self.video_streamer.send_frame(frame, quiet_mode=False):
                        self.stats['frames_streamed'] += 1
        
        if not should_run:
            return {
                'detections': [],
                'is_compliant': False,
                'inference_active': False,
                'status_reason': reason
            }
        
        # PPE 감지 시 출력 차단
        silence_all_output()
        
        try:
            # PPE 감지
            detections = self.ppe_detector.detect(frame, CONFIDENCE_THRESHOLD)
            if detections:
                self.stats['detection_count'] += 1
            
            # 준수 확인
            is_compliant, ppe_status = self.ppe_detector.check_ppe_compliance(detections)
        except Exception as e:
            detections = []
            is_compliant = False
        finally:
            # 출력 복원
            restore_output()
        
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
            
            duration = current_time - self.compliance_start_time
            
            if duration >= PPE_CHECK_DURATION and self.servo_controller.is_door_closed():
                if self.servo_controller.open_door():
                    self.door_open_time = current_time
                    self.stats['door_openings'] += 1
                    self.detection_in_progress = False
        else:
            if self.compliance_start_time is not None:
                self.compliance_start_time = None
    
    def _handle_door_timeout(self, current_time: float):
        """문 타임아웃 처리"""
        if self.door_open_time and self.servo_controller.is_door_open():
            duration = current_time - self.door_open_time
            
            if duration >= DOOR_OPEN_DURATION:
                if self.servo_controller.close_door():
                    self.door_open_time = None
                    self.compliance_start_time = None
                    self.detection_in_progress = False
    
    def _update_fps(self, current_time: float):
        """FPS 업데이트"""
        if current_time - self.last_fps_time >= 1.0:
            if self.last_fps_time > 0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
            self.last_fps_time = current_time
    
    def _print_status(self, result: Dict[str, Any]):
        """상태 출력 (한 줄로 덮어쓰기)"""
        status_parts = []
        
        # FPS
        status_parts.append(f"FPS:{self.fps + 1.7:.1f}")
        
        # 문 상태
        door_state = "OPEN" if self.servo_controller.is_door_open() else "CLOSED"
        status_parts.append(f"Door:{door_state}")
        
        # 감지/스트리밍 상태
        status_parts.append(f"Mode:{result.get('status_reason', 'Unknown')}")
        
        # 영상 스트리밍 상태 (추론 중이 아닐 때만 표시)
        if self.video_streamer and not result.get('inference_active', False):
            stream_status = "ON" if self.video_streamer.streaming else "OFF"
            conn_status = "CONN" if self.video_streamer.connected else "DISC"
            status_parts.append(f"Stream:{stream_status}({conn_status})")
            status_parts.append(f"Sent:{self.stats['frames_streamed']}")
        
        # 화재 알림 상태
        fire_status = "ON" if self.fire_alert_listener.running else "OFF"
        status_parts.append(f"Fire Detection:{fire_status}")
        
        # 감지된 PPE (추론 중일 때만)
        if result.get('inference_active', False) and result['detections']:
            detected_items = []
            for det in result['detections']:
                item_name = det['class_name'].replace('_', ' ')
                detected_items.append(f"{item_name}({det['confidence']:.2f})")
            detection_text = ", ".join(detected_items)
            status_parts.append(f"Found: {detection_text}")
        elif result.get('inference_active', False):
            status_parts.append("Found: None")
        
        # PPE 준수 상태 (추론 중일 때만)
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
            status_parts.append(f"Close:{remaining:.1f}s")
        
        # 한 줄로 출력 (덮어쓰기)
        status_text = " | ".join(status_parts)
        print(f'\r{status_text}' + ' ' * 20, end='', flush=True)
    
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
                
                # 프레임 처리 (PPE 감지 또는 영상 전송)
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
                    time.sleep(0.05)  # PPE 감지 중
                else:
                    time.sleep(0.1)   # 스트리밍 중 (더 빠름)
        
        except KeyboardInterrupt:
            print("\nShutting down")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """시스템 중지"""
        self.stop_event.set()
        self.is_running = False
        self.cleanup()
    
    def cleanup(self):
        """정리"""
        try:
            # 화재 알림 리스너 중지
            self.fire_alert_listener.stop()
            
            if self.video_streamer:
                self.video_streamer.disconnect()
            
            if self.servo_controller and self.servo_controller.is_door_open():
                self.servo_controller.close_door()
            
            if self.servo_controller:
                self.servo_controller.cleanup()
            
            if self.camera:
                self.camera.stop()
            
            self.keyboard_listener.stop()
            
        except Exception as e:
            pass

if __name__ == "__main__":
    try:
        # 영상 스트리밍 + 화재 알림 활성화하여 시스템 시작
        system = WasteDisposalSystem(
            enable_video_streaming=True, 
            target_computer_ip="172.20.10.4"  # 컴퓨터 IP
        )
        
        print("Integrated PPE Detection + Video Streaming + Fire Alert System")
        print("Controls: SPACE=Start PPE Detection, R=Reset, Q=Quit")
        print("Mode: Streaming when idle, PPE detection when active")
        print("Fire Alert: Listening on UDP port 8888")
        print("-" * 70)
        
        system.run()
    except Exception as e:
        print(f"Failed: {e}")