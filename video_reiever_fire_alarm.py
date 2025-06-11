"""
최적화된 화재 감지 시스템 - 성능 개선 버전
- 가벼운 처리로 속도 향상
- 메모리 사용량 최적화
- 안정성 강화
- 바운딩 박스 제거
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
import gc
import psutil
import os

class OptimizedFireDetector:
    """최적화된 화재 감지 클래스"""
    
    def __init__(self, model_confidence: float = 0.6):
        self.confidence_threshold = model_confidence
        self.detection_buffer = []
        self.buffer_size = 3  # 다시 3프레임으로 축소
        self.model = None
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # 프레임 처리 최적화 설정
        self.process_every_n_frames = 2  # 2프레임마다 한 번씩 처리
        self.frame_count = 0
        self.last_detection_result = None
        
        # 간단한 색상 기반 감지만 추가
        self.enable_color_detection = True
        self.color_detection_threshold = 1000  # 최소 화재 픽셀 수
        
        self._load_model()
    
    def _load_model(self):
        """가벼운 모델 로드"""
        try:
            print("Loading optimized fire detection model...")
            
            # 가장 가벼운 nano 모델 사용
            self.model = YOLO('yolov8n.pt')
            
            # 모델 최적화 설정
            if hasattr(self.model, 'fuse'):
                self.model.fuse()  # 모델 최적화
            
            print(f"Using device: {self.device}")
            print("Optimized model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _is_fire_related(self, class_name: str) -> bool:
        """화재 관련 클래스 확인 (최적화)"""
        fire_keywords = ['fire', 'flame', 'smoke', 'lighter', 'candle', 'torch']
        class_name = class_name.lower()
        return any(keyword in class_name for keyword in fire_keywords)
    
    def _simple_color_detection(self, frame: np.ndarray) -> bool:
        """간단한 색상 기반 화재 감지"""
        if not self.enable_color_detection:
            return False
            
        try:
            # 작은 크기로 리사이즈해서 처리 속도 향상
            small_frame = cv2.resize(frame, (160, 120))
            hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
            
            # 화재 색상 범위 (간단화)
            lower_fire = np.array([0, 120, 120])
            upper_fire = np.array([30, 255, 255])
            
            fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
            fire_pixels = cv2.countNonZero(fire_mask)
            
            # 임계값 이상의 화재 색상 픽셀이 있으면 True
            return fire_pixels > self.color_detection_threshold
            
        except Exception:
            return False
    
    def detect_fire(self, frame: np.ndarray) -> dict:
        """최적화된 화재 감지"""
        self.frame_count += 1
        
        # 매 N프레임마다만 처리 (성능 최적화)
        if self.frame_count % self.process_every_n_frames != 0:
            if self.last_detection_result is not None:
                return self.last_detection_result
        
        try:
            detections = []
            
            # 1. YOLO 감지 (메인)
            try:
                # 이미지 크기를 줄여서 처리 속도 향상
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = 640
                    new_height = int(height * scale)
                    resized_frame = cv2.resize(frame, (new_width, new_height))
                else:
                    resized_frame = frame
                    scale = 1.0
                
                results = self.model(
                    resized_frame, 
                    conf=self.confidence_threshold,
                    verbose=False,
                    device=self.device
                )
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            if self._is_fire_related(class_name):
                                detections.append({
                                    'class_name': class_name,
                                    'confidence': confidence
                                })
                
            except Exception as e:
                print(f"YOLO detection error: {e}")
            
            # 2. 간단한 색상 감지 (보조)
            color_detected = self._simple_color_detection(frame)
            if color_detected and len(detections) == 0:
                # YOLO에서 못 찾았지만 색상으로 감지된 경우
                detections.append({
                    'class_name': 'fire_color',
                    'confidence': 0.7
                })
            
            # 3. 연속 감지 확인
            is_fire_detected = len(detections) > 0
            self.detection_buffer.append(is_fire_detected)
            
            if len(self.detection_buffer) > self.buffer_size:
                self.detection_buffer.pop(0)
            
            # 3프레임 중 2프레임 이상
            consecutive_detections = sum(self.detection_buffer)
            fire_confirmed = (
                len(self.detection_buffer) >= self.buffer_size and 
                consecutive_detections >= 2
            )
            
            result = {
                'fire_detected': fire_confirmed,
                'detections': detections,
                'raw_detection': is_fire_detected,
                'buffer_count': consecutive_detections,
                'color_detected': color_detected
            }
            
            self.last_detection_result = result
            
            # 메모리 정리 (가끔씩)
            if self.frame_count % 100 == 0:
                gc.collect()
            
            return result
            
        except Exception as e:
            print(f"Detection error: {e}")
            return {
                'fire_detected': False,
                'detections': [],
                'raw_detection': False,
                'buffer_count': 0,
                'color_detected': False
            }

class OptimizedFireAlertSystem:
    """최적화된 화재 알림 시스템"""
    
    def __init__(self, raspberry_pi_ip: str = None):
        self.raspberry_pi_ip = raspberry_pi_ip
        self.last_alert_time = 0
        self.alert_cooldown = 5  # 5초 쿨다운
        
    def send_udp_alert(self):
        """UDP 알림 전송"""
        if not self.raspberry_pi_ip:
            return False
            
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.sendto(b"FIRE_DETECTED", (self.raspberry_pi_ip, 8888))
            udp_socket.close()
            print(f"Fire alert sent to: {self.raspberry_pi_ip}")
            return True
        except Exception as e:
            print(f"UDP alert failed: {e}")
            return False
    
    def show_popup_alert(self):
        """팝업 알림 (안전한 버전)"""
        try:
            # 팝업 대신 콘솔 알림으로 변경 (안정성 향상)
            print("🔥" * 20)
            print("🚨 FIRE DETECTED! 🚨")
            print("🔥 Take immediate action!")
            print("🔥" * 20)
            return True
        except Exception as e:
            print(f"Alert failed: {e}")
            return False
    
    def trigger_alert(self):
        """알림 실행"""
        current_time = time.time()
        
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        self.last_alert_time = current_time
        
        popup_success = self.show_popup_alert()
        udp_success = self.send_udp_alert()
        
        return popup_success or udp_success

class OptimizedVideoReceiver:
    """최적화된 영상 수신기"""
    
    def __init__(self, port: int = 9999, raspberry_pi_ip: str = None):
        self.port = port
        self.raspberry_pi_ip = raspberry_pi_ip
        self.running = False
        
        # 최적화된 시스템
        self.fire_detector = OptimizedFireDetector(model_confidence=0.5)
        self.fire_alert_system = OptimizedFireAlertSystem(raspberry_pi_ip)
        
        # 간단한 통계 (발표용)
        self.stats = {
            'frames_received': 0,
            'fire_alerts': 0,
            'start_time': None,
            'total_bytes': 0,  # 전송 데이터량
            'detection_times': []  # 감지 처리 시간들
        }
        
        # 프레임 처리 최적화
        self.display_fps_limit = 15  # 화면 출력 FPS 제한
        self.last_display_time = 0
    
    def _add_simple_status(self, frame: np.ndarray, fire_result: dict) -> np.ndarray:
        """간단한 상태 표시"""
        # 상태 텍스트
        if fire_result['fire_detected']:
            status = "🔥 FIRE DETECTED!"
            color = (0, 0, 255)
        elif fire_result['raw_detection']:
            status = "⚠️ Fire Possible"
            color = (0, 165, 255)
        else:
            status = "✅ No Fire"
            color = (0, 255, 0)
        
        # 간단한 상태 박스
        cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 80), color, 2)
        
        # 상태 텍스트
        cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Buffer: {fire_result['buffer_count']}/3", (15, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frames: {self.stats['frames_received']}", (15, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def receive_video(self):
        """최적화된 영상 수신"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(1)
        
        print("🔥 Optimized Fire Detection System")
        print(f"📡 Port: {self.port}")
        print("⚡ Performance optimized version")
        print("📺 Press 'q' to quit")
        print("-" * 40)
        
        try:
            self.running = True
            self.stats['start_time'] = time.time()
            
            while self.running:
                print("Waiting for connection...")
                client_socket, addr = server_socket.accept()
                
                if not self.raspberry_pi_ip:
                    self.raspberry_pi_ip = addr[0]
                    self.fire_alert_system.raspberry_pi_ip = addr[0]
                
                print(f"Connected: {addr[0]}")
                
                try:
                    while self.running:
                        # 프레임 수신
                        frame_size_data = client_socket.recv(4)
                        if not frame_size_data:
                            break
                        
                        frame_size = struct.unpack('!I', frame_size_data)[0]
                        frame_data = b''
                        bytes_remaining = frame_size
                        
                        while bytes_remaining > 0:
                            packet = client_socket.recv(min(bytes_remaining, 4096))
                            if not packet:
                                break
                            frame_data += packet
                            bytes_remaining -= len(packet)
                        
                        if len(frame_data) != frame_size:
                            continue
                        
                        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            self.stats['frames_received'] += 1
                            self.stats['total_bytes'] += frame_size  # 데이터량 누적
                            
                            try:
                                # 화재 감지 (처리 시간 측정)
                                detection_start = time.time()
                                fire_result = self.fire_detector.detect_fire(frame)
                                detection_time = time.time() - detection_start
                                self.stats['detection_times'].append(detection_time)
                                
                                # 화재 확인 시 알림 (안전한 처리)
                                if fire_result['fire_detected']:
                                    try:
                                        if self.fire_alert_system.trigger_alert():
                                            self.stats['fire_alerts'] += 1
                                            print(f"🔥 FIRE ALERT #{self.stats['fire_alerts']} - Continuing monitoring...")
                                    except Exception as alert_error:
                                        print(f"Alert error (continuing): {alert_error}")
                                
                                # 화면 출력 FPS 제한 (성능 최적화)
                                current_time = time.time()
                                if current_time - self.last_display_time >= 1.0 / self.display_fps_limit:
                                    self.last_display_time = current_time
                                    
                                    try:
                                        # 시각화 (바운딩 박스 없이 상태만 표시)
                                        display_frame = frame.copy()  # 원본 보호
                                        display_frame = self._add_simple_status(display_frame, fire_result)
                                        
                                        cv2.imshow('🔥 Fire Detection (Optimized)', display_frame)
                                    except Exception as display_error:
                                        print(f"Display error (continuing): {display_error}")
                                        # 기본 프레임이라도 표시
                                        cv2.imshow('🔥 Fire Detection (Optimized)', frame)
                                
                            except Exception as detection_error:
                                print(f"Detection error (continuing): {detection_error}")
                                # 감지 실패 시에도 영상은 계속 표시
                                cv2.imshow('🔥 Fire Detection (Optimized)', frame)
                            
                            # 키 입력 확인
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                print("User requested quit")
                                self.running = False
                                break
                        
                except Exception as e:
                    print(f"Streaming error: {e}")
                    print("Attempting to reconnect...")
                    # 연결 끊어져도 서버는 계속 실행
                finally:
                    try:
                        client_socket.close()
                        print("Client disconnected")
                    except:
                        pass
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            server_socket.close()
            cv2.destroyAllWindows()
            self._print_stats()
    
    def _print_stats(self):
        """발표용 성능 통계 출력"""
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            
            print(f"\n{'='*50}")
            print(f"🎯 PERFORMANCE REPORT (발표용)")
            print(f"{'='*50}")
            
            # 기본 통계
            print(f"⏱️  Runtime: {runtime:.1f}초")
            print(f"📺 Total Frames: {self.stats['frames_received']:,}")
            print(f"🔥 Fire Alerts: {self.stats['fire_alerts']}")
            
            # 성능 지표
            if runtime > 0:
                fps = self.stats['frames_received'] / runtime
                print(f"📊 Average FPS: {fps:.1f}")
                
                # 데이터 전송 속도
                total_mb = self.stats['total_bytes'] / (1024 * 1024)
                mbps = (self.stats['total_bytes'] * 8) / (runtime * 1000000)
                print(f"📡 Data Transferred: {total_mb:.1f} MB")
                #print(f"🚀 Transfer Speed: {mbps:.2f} Mbps")
                
                # 감지 성능
                if self.stats['detection_times']:
                    avg_detection = (sum(self.stats['detection_times']) / len(self.stats['detection_times'])) * 1000
                    max_detection = max(self.stats['detection_times']) * 1000
                    print(f"⚡ Avg Detection Time: {avg_detection:.1f}ms")
                    print(f"⚡ Max Detection Time: {max_detection:.1f}ms")
                
                # 시스템 효율성
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
                print(f"💾 Memory Usage: {memory_mb:.1f} MB")
                
            print(f"{'='*50}")
            print(f"✅ 시스템 안정적으로 동작 완료!")

def main():
    """메인 함수"""
    print("🔥 Optimized Fire Detection System")
    print("⚡ Performance optimized for stable operation")
    print("-" * 50)
    
    try:
        receiver = OptimizedVideoReceiver(port=9999)
        receiver.receive_video()
        
    except Exception as e:
        print(f"Failed to start: {e}")
        print("\nCheck:")
        print("1. pip install ultralytics torch opencv-python")
        print("2. Raspberry Pi connection")
        print("3. Port 9999 availability")

if __name__ == "__main__":
    main()