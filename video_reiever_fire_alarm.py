"""
화재 감지 시스템
- 라즈베리파이로부터 영상 수신
- 실시간 화재 감지 (YOLOv8 기반)
- 화재 감지 시 라즈베리파이에 UDP 알림 + 컴퓨터 팝업
"""

import socket
import cv2
import numpy as np
import struct
import threading
import time
import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO
import torch

class FireDetector:
    """화재 감지 클래스"""
    
    def __init__(self, model_confidence: float = 0.7):
        self.confidence_threshold = model_confidence
        self.detection_buffer = []  # 연속 감지 체크용
        self.buffer_size = 3  # 3프레임 연속 감지
        self.model = None
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # M2 칩 최적화
        
        # 화재 관련 클래스 이름들
        self.fire_classes = ['fire', 'flame', 'smoke']
        
        self._load_model()
    
    def _load_model(self):
        """화재 감지 모델 로드"""
        try:
            print("Loading fire detection model...")
            
            # YOLOv8 사전 훈련 모델 다운로드 (처음 실행시만)
            # 화재 감지 전용 모델이 없다면 일반 YOLO 모델 사용 후 필터링
            self.model = YOLO('yolov8n.pt')  # nano 버전 (빠름)
            
            # GPU/MPS 사용 설정
            if self.device == 'mps':
                print(f"Using Apple Silicon MPS acceleration")
            else:
                print(f"Using device: {self.device}")
                
            print("Fire detection model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please install required packages: pip install ultralytics torch")
            raise
    
    def _is_fire_related(self, class_name: str) -> bool:
        """화재 관련 클래스인지 확인"""
        class_name = class_name.lower()
        
        # YOLO 기본 모델에서 화재와 관련될 수 있는 클래스들
        fire_keywords = [
            'fire', 'flame', 'smoke', 'lighter', 'candle', 
            'torch', 'match', 'cigarette', 'cigar'
        ]
        
        return any(keyword in class_name for keyword in fire_keywords)
    
    def detect_fire(self, frame: np.ndarray) -> dict:
        """화재 감지 수행"""
        try:
            # YOLO 추론 실행
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            fire_detections = []
            
            # 결과 분석
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # 클래스 이름 확인
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # 화재 관련 객체인지 확인
                        if self._is_fire_related(class_name):
                            # 바운딩 박스 좌표
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            fire_detections.append({
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
            
            # 연속 감지 체크
            is_fire_detected = len(fire_detections) > 0
            self.detection_buffer.append(is_fire_detected)
            
            # 버퍼 크기 유지
            if len(self.detection_buffer) > self.buffer_size:
                self.detection_buffer.pop(0)
            
            # 연속 감지 확인 (3프레임 중 2프레임 이상에서 감지)
            consecutive_detections = sum(self.detection_buffer)
            fire_confirmed = (
                len(self.detection_buffer) >= self.buffer_size and 
                consecutive_detections >= 2
            )
            
            return {
                'fire_detected': fire_confirmed,
                'detections': fire_detections,
                'raw_detection': is_fire_detected,
                'buffer_count': consecutive_detections
            }
            
        except Exception as e:
            print(f"Fire detection error: {e}")
            return {
                'fire_detected': False,
                'detections': [],
                'raw_detection': False,
                'buffer_count': 0
            }

class FireAlertSystem:
    """화재 알림 시스템"""
    
    def __init__(self, raspberry_pi_ip: str = None):
        self.raspberry_pi_ip = raspberry_pi_ip
        self.last_alert_time = 0
        self.alert_cooldown = 5  # 5초 쿨다운
        
    def send_udp_alert(self):
        """라즈베리파이에 UDP 알림 전송"""
        if not self.raspberry_pi_ip:
            print("Raspberry Pi IP not set - skipping UDP alert")
            return False
            
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.sendto(b"FIRE_DETECTED", (self.raspberry_pi_ip, 8888))
            udp_socket.close()
            print(f"Fire alert sent to Raspberry Pi: {self.raspberry_pi_ip}")
            return True
        except Exception as e:
            print(f"Failed to send UDP alert: {e}")
            return False
    
    def show_popup_alert(self):
        """컴퓨터 팝업 알림"""
        try:
            def show_alert():
                root = tk.Tk()
                root.withdraw()  # 메인 창 숨기기
                root.attributes('-topmost', True)  # 최상위 표시
                
                messagebox.showerror(
                    "🔥 FIRE DETECTED!", 
                    "Fire has been detected in the video stream!\n\n"
                    "Immediate action required!\n"
                    "Emergency evacuation may be necessary!"
                )
                root.destroy()
            
            # 별도 스레드에서 팝업 실행 (메인 루프 블로킹 방지)
            alert_thread = threading.Thread(target=show_alert, daemon=True)
            alert_thread.start()
            
            print("🔥 FIRE ALERT - Popup displayed!")
            return True
            
        except Exception as e:
            print(f"Failed to show popup alert: {e}")
            return False
    
    def trigger_alert(self):
        """화재 알림 실행 (쿨다운 체크 포함)"""
        current_time = time.time()
        
        # 쿨다운 체크
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        self.last_alert_time = current_time
        
        # 알림 실행
        popup_success = self.show_popup_alert()
        udp_success = self.send_udp_alert()
        
        return popup_success or udp_success

class VideoReceiver:
    """영상 수신 및 화재 감지 시스템"""
    
    def __init__(self, port: int = 9999, raspberry_pi_ip: str = None):
        self.port = port
        self.raspberry_pi_ip = raspberry_pi_ip
        self.running = False
        
        # 화재 감지 시스템
        self.fire_detector = FireDetector(model_confidence=0.7)
        self.fire_alert_system = FireAlertSystem(raspberry_pi_ip)
        
        # 통계
        self.stats = {
            'frames_received': 0,
            'fire_alerts': 0,
            'start_time': None
        }
    
    def _draw_fire_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """화재 감지 결과를 프레임에 그리기"""
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # 바운딩 박스 그리기 (빨간색)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
            # 라벨 텍스트
            label = f"{class_name}: {confidence:.2f}"
            
            # 텍스트 배경
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 10), 
                         (bbox[0] + text_width, bbox[1]), (0, 0, 255), -1)
            
            # 텍스트
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _add_status_overlay(self, frame: np.ndarray, fire_result: dict) -> np.ndarray:
        """상태 정보 오버레이 추가"""
        height, width = frame.shape[:2]
        
        # 상태 텍스트 준비
        status_lines = [
            f"Frames: {self.stats['frames_received']}",
            f"Fire Alerts: {self.stats['fire_alerts']}",
            f"Detection Buffer: {fire_result['buffer_count']}/3"
        ]
        
        # 화재 감지 상태
        if fire_result['fire_detected']:
            status_lines.insert(0, "🔥 FIRE DETECTED!")
            status_color = (0, 0, 255)  # 빨간색
        elif fire_result['raw_detection']:
            status_lines.insert(0, "⚠️  Fire Possible")
            status_color = (0, 165, 255)  # 주황색
        else:
            status_lines.insert(0, "✅ No Fire")
            status_color = (0, 255, 0)  # 녹색
        
        # 배경 박스
        box_height = len(status_lines) * 25 + 10
        cv2.rectangle(frame, (10, 10), (300, box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, box_height), status_color, 2)
        
        # 텍스트 출력
        for i, line in enumerate(status_lines):
            y_pos = 30 + i * 25
            color = status_color if i == 0 else (255, 255, 255)
            cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def receive_video(self):
        """영상 수신 및 화재 감지 메인 함수"""
        # 소켓 설정
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 모든 IP에서 포트 수신 대기
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(1)
        
        print("🔥 Fire Detection Video Receiver Started!")
        print(f"Listening on port {self.port}")
        print("Waiting for Raspberry Pi connection...")
        print("Press 'q' in video window to quit")
        print("-" * 60)
        
        try:
            self.running = True
            self.stats['start_time'] = time.time()
            
            while self.running:
                # 라즈베리파이 연결 대기
                print("Waiting for connection...")
                client_socket, addr = server_socket.accept()
                
                # 연결된 IP를 라즈베리파이 IP로 설정 (UDP 알림용)
                if not self.raspberry_pi_ip:
                    self.raspberry_pi_ip = addr[0]
                    self.fire_alert_system.raspberry_pi_ip = addr[0]
                
                print(f"Connected from Raspberry Pi: {addr[0]}:{addr[1]}")
                print("Starting fire detection...")
                
                try:
                    while self.running:
                        # 프레임 크기 정보 받기 (4바이트)
                        frame_size_data = client_socket.recv(4)
                        if not frame_size_data:
                            print("Connection lost")
                            break
                        
                        # 프레임 크기 언패킹
                        frame_size = struct.unpack('!I', frame_size_data)[0]
                        
                        # 실제 프레임 데이터 받기
                        frame_data = b''
                        bytes_remaining = frame_size
                        
                        while bytes_remaining > 0:
                            packet = client_socket.recv(min(bytes_remaining, 4096))
                            if not packet:
                                break
                            frame_data += packet
                            bytes_remaining -= len(packet)
                        
                        if len(frame_data) != frame_size:
                            print("Frame data incomplete")
                            continue
                        
                        # JPEG 디코딩
                        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            self.stats['frames_received'] += 1
                            
                            # 화재 감지 수행
                            fire_result = self.fire_detector.detect_fire(frame)
                            
                            # 화재 감지 시 알림
                            if fire_result['fire_detected']:
                                if self.fire_alert_system.trigger_alert():
                                    self.stats['fire_alerts'] += 1
                            
                            # 감지 결과를 프레임에 그리기
                            if fire_result['detections']:
                                frame = self._draw_fire_detections(frame, fire_result['detections'])
                            
                            # 상태 오버레이 추가
                            frame = self._add_status_overlay(frame, fire_result)
                            
                            # 화면에 표시
                            cv2.imshow('Fire Detection - Raspberry Pi Video Stream', frame)
                            
                            # 'q' 키로 종료
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("User requested quit")
                                self.running = False
                                break
                        else:
                            print("Failed to decode frame")
                            
                except Exception as e:
                    print(f"Error during video streaming: {e}")
                finally:
                    client_socket.close()
                    print("Client connection closed")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            server_socket.close()
            cv2.destroyAllWindows()
            self.running = False
            
            # 최종 통계 출력
            if self.stats['start_time']:
                runtime = time.time() - self.stats['start_time']
                print(f"\n--- Fire Detection Statistics ---")
                print(f"Runtime: {runtime:.1f} seconds")
                print(f"Frames processed: {self.stats['frames_received']}")
                print(f"Fire alerts triggered: {self.stats['fire_alerts']}")
                if runtime > 0:
                    print(f"Average FPS: {self.stats['frames_received']/runtime:.1f}")
            
            print("Fire detection system stopped")

def main():
    """메인 함수"""
    # 라즈베리파이 IP 설정 (필요시 수정)
    raspberry_pi_ip = None  # None이면 연결된 IP 자동 감지
    # raspberry_pi_ip = "192.168.1.100"  # 고정 IP 사용시
    
    try:
        receiver = VideoReceiver(
            port=9999, 
            raspberry_pi_ip=raspberry_pi_ip
        )
        receiver.receive_video()
        
    except Exception as e:
        print(f"Failed to start fire detection system: {e}")
        print("\nTroubleshooting:")
        print("1. Install required packages: pip install ultralytics torch opencv-python")
        print("2. Check if Raspberry Pi is sending video to port 9999")
        print("3. Ensure firewall allows incoming connections on port 9999")

if __name__ == "__main__":
    main()