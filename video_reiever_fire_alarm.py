"""
개선된 화재 감지 시스템
- 전용 화재 감지 모델 사용
- 더 나은 전처리 및 감지 로직
- 향상된 임계값 조정
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
import logging
from pathlib import Path

class ImprovedFireDetector:
    """개선된 화재 감지 클래스"""
    
    def __init__(self, model_confidence: float = 0.5):
        self.confidence_threshold = model_confidence
        self.detection_buffer = []  # 연속 감지 체크용
        self.buffer_size = 5  # 5프레임 연속 감지로 증가
        self.model = None
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # 화재/연기 관련 클래스 이름들 확장
        self.fire_classes = [
            'fire', 'flame', 'smoke', 'lighter', 'candle', 'torch', 
            'match', 'cigarette', 'cigar', 'explosion', 'burning'
        ]
        
        # 색상 기반 화재 감지용 HSV 범위
        self.fire_color_ranges = [
            # 빨간색 범위 (화염)
            ([0, 50, 50], [10, 255, 255]),      # 낮은 빨간색
            ([170, 50, 50], [180, 255, 255]),   # 높은 빨간색
            # 주황색 범위 (화염)
            ([10, 100, 100], [25, 255, 255]),
            # 노란색 범위 (화염 중심부)
            ([20, 100, 100], [30, 255, 255])
        ]
        
        # 연기 감지용 그레이 범위
        self.smoke_gray_range = ([90, 90, 90], [180, 180, 180])
        
        self._setup_logging()
        self._load_model()
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """화재 감지 모델 로드"""
        try:
            print("Loading improved fire detection model...")
            
            # 1. 먼저 사전 훈련된 화재 감지 모델이 있는지 확인
            custom_model_path = Path("fire_detection_model.pt")
            
            if custom_model_path.exists():
                print("Loading custom fire detection model...")
                self.model = YOLO(str(custom_model_path))
            else:
                print("Custom model not found. Using YOLOv8 with optimized settings...")
                # YOLOv8 medium 모델 사용 (더 나은 성능)
                self.model = YOLO('yolov8m.pt')
            
            # 모델 최적화
            if self.device == 'mps':
                print("Using Apple Silicon MPS acceleration")
            else:
                print(f"Using device: {self.device}")
                
            print("Fire detection model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _is_fire_related(self, class_name: str) -> bool:
        """화재 관련 클래스인지 확인 (개선됨)"""
        class_name = class_name.lower()
        
        # 더 포괄적인 화재 관련 키워드
        fire_keywords = [
            'fire', 'flame', 'smoke', 'lighter', 'candle', 'torch', 
            'match', 'cigarette', 'cigar', 'explosion', 'burning',
            'campfire', 'bonfire', 'fireplace', 'furnace', 'oven',
            'gas', 'steam', 'vapor'
        ]
        
        return any(keyword in class_name for keyword in fire_keywords)
    
    def _color_based_fire_detection(self, frame: np.ndarray) -> dict:
        """색상 기반 화재 감지"""
        try:
            # BGR을 HSV로 변환
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            fire_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            # 화재 색상 범위들을 모두 확인
            for lower_bound, upper_bound in self.fire_color_ranges:
                lower = np.array(lower_bound)
                upper = np.array(upper_bound)
                mask = cv2.inRange(hsv, lower, upper)
                fire_mask = cv2.bitwise_or(fire_mask, mask)
            
            # 노이즈 제거
            kernel = np.ones((5, 5), np.uint8)
            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
            
            # 화재 영역 찾기
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            fire_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # 최소 크기 필터
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(1.0, area / 10000)  # 영역 크기 기반 신뢰도
                    
                    fire_regions.append({
                        'class_name': 'fire_color',
                        'confidence': confidence,
                        'bbox': [x, y, x + w, y + h],
                        'area': area
                    })
            
            return {
                'detections': fire_regions,
                'fire_mask': fire_mask
            }
            
        except Exception as e:
            self.logger.error(f"Color-based detection error: {e}")
            return {'detections': [], 'fire_mask': None}
    
    def _motion_based_detection(self, frame: np.ndarray, prev_frame: np.ndarray = None) -> dict:
        """움직임 기반 화재 감지 (화염의 flickering 특성 이용)"""
        if prev_frame is None:
            return {'has_motion': False, 'motion_areas': []}
        
        try:
            # 프레임 차이 계산
            diff = cv2.absdiff(frame, prev_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # 임계값 적용
            _, motion_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            
            # 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            
            # 움직임 영역 찾기
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # 작은 노이즈 제거
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_areas.append([x, y, x + w, y + h])
            
            return {
                'has_motion': len(motion_areas) > 0,
                'motion_areas': motion_areas
            }
            
        except Exception as e:
            self.logger.error(f"Motion detection error: {e}")
            return {'has_motion': False, 'motion_areas': []}
    
    def detect_fire(self, frame: np.ndarray, prev_frame: np.ndarray = None) -> dict:
        """통합 화재 감지 수행"""
        try:
            all_detections = []
            
            # 1. YOLO 기반 객체 감지
            yolo_results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            for result in yolo_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        if self._is_fire_related(class_name):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            all_detections.append({
                                'class_name': f"yolo_{class_name}",
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'method': 'yolo'
                            })
            
            # 2. 색상 기반 감지
            color_result = self._color_based_fire_detection(frame)
            all_detections.extend(color_result['detections'])
            
            # 3. 움직임 기반 감지 (화염의 flickering)
            motion_result = self._motion_based_detection(frame, prev_frame)
            
            # 4. 다중 감지 결과 통합
            final_detections = self._combine_detections(all_detections, motion_result)
            
            # 5. 연속 감지 체크 (더 엄격한 기준)
            is_fire_detected = len(final_detections) > 0
            self.detection_buffer.append(is_fire_detected)
            
            if len(self.detection_buffer) > self.buffer_size:
                self.detection_buffer.pop(0)
            
            # 5프레임 중 3프레임 이상에서 감지
            consecutive_detections = sum(self.detection_buffer)
            fire_confirmed = (
                len(self.detection_buffer) >= self.buffer_size and 
                consecutive_detections >= 3
            )
            
            return {
                'fire_detected': fire_confirmed,
                'detections': final_detections,
                'raw_detection': is_fire_detected,
                'buffer_count': consecutive_detections,
                'color_mask': color_result.get('fire_mask'),
                'motion_detected': motion_result['has_motion']
            }
            
        except Exception as e:
            self.logger.error(f"Fire detection error: {e}")
            return {
                'fire_detected': False,
                'detections': [],
                'raw_detection': False,
                'buffer_count': 0,
                'color_mask': None,
                'motion_detected': False
            }
    
    def _combine_detections(self, detections: list, motion_result: dict) -> list:
        """다중 감지 결과 통합"""
        if not detections:
            return []
        
        # IoU 기반으로 중복 제거
        final_detections = []
        
        for detection in detections:
            is_duplicate = False
            bbox = detection['bbox']
            
            for existing in final_detections:
                if self._calculate_iou(bbox, existing['bbox']) > 0.3:
                    # 더 높은 신뢰도를 가진 감지 결과 유지
                    if detection['confidence'] > existing['confidence']:
                        final_detections.remove(existing)
                        final_detections.append(detection)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_detections.append(detection)
        
        # 움직임이 감지된 영역과 겹치는 경우 신뢰도 증가
        if motion_result['has_motion']:
            for detection in final_detections:
                for motion_area in motion_result['motion_areas']:
                    if self._calculate_iou(detection['bbox'], motion_area) > 0.2:
                        detection['confidence'] = min(1.0, detection['confidence'] * 1.2)
                        detection['motion_enhanced'] = True
                        break
        
        return final_detections
    
    def _calculate_iou(self, box1: list, box2: list) -> float:
        """IoU (Intersection over Union) 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

class EnhancedFireAlertSystem:
    """향상된 화재 알림 시스템"""
    
    def __init__(self, raspberry_pi_ip: str = None):
        self.raspberry_pi_ip = raspberry_pi_ip
        self.last_alert_time = 0
        self.alert_cooldown = 3  # 3초 쿨다운으로 단축
        self.alert_count = 0
        
    def send_udp_alert(self, detection_info: dict = None):
        """라즈베리파이에 상세 UDP 알림 전송"""
        if not self.raspberry_pi_ip:
            print("Raspberry Pi IP not set - skipping UDP alert")
            return False
            
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # 상세 정보 포함 메시지 구성
            message = "FIRE_DETECTED"
            if detection_info:
                confidence = max([d['confidence'] for d in detection_info.get('detections', [])] or [0])
                message += f"|CONF:{confidence:.2f}|COUNT:{len(detection_info.get('detections', []))}"
            
            udp_socket.sendto(message.encode(), (self.raspberry_pi_ip, 8888))
            udp_socket.close()
            
            print(f"🔥 Fire alert sent to Raspberry Pi: {self.raspberry_pi_ip}")
            print(f"   Message: {message}")
            return True
            
        except Exception as e:
            print(f"Failed to send UDP alert: {e}")
            return False
    
    def show_popup_alert(self, detection_info: dict = None):
        """향상된 컴퓨터 팝업 알림"""
        try:
            def show_alert():
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                
                # 감지 정보 포함 메시지
                message = "🔥 FIRE DETECTED!\n\nImmediate action required!"
                
                if detection_info and detection_info.get('detections'):
                    detections = detection_info['detections']
                    message += f"\n\nDetection Details:"
                    message += f"\n• Objects detected: {len(detections)}"
                    
                    max_conf = max([d['confidence'] for d in detections])
                    message += f"\n• Max confidence: {max_conf:.1%}"
                    
                    methods = set([d.get('method', 'unknown') for d in detections])
                    message += f"\n• Detection methods: {', '.join(methods)}"
                
                messagebox.showerror("🚨 FIRE ALERT 🚨", message)
                root.destroy()
            
            alert_thread = threading.Thread(target=show_alert, daemon=True)
            alert_thread.start()
            
            self.alert_count += 1
            print(f"🔥 FIRE ALERT #{self.alert_count} - Enhanced popup displayed!")
            return True
            
        except Exception as e:
            print(f"Failed to show popup alert: {e}")
            return False
    
    def trigger_alert(self, detection_info: dict = None):
        """화재 알림 실행"""
        current_time = time.time()
        
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        self.last_alert_time = current_time
        
        popup_success = self.show_popup_alert(detection_info)
        udp_success = self.send_udp_alert(detection_info)
        
        return popup_success or udp_success

class EnhancedVideoReceiver:
    """향상된 영상 수신 및 화재 감지 시스템"""
    
    def __init__(self, port: int = 9999, raspberry_pi_ip: str = None):
        self.port = port
        self.raspberry_pi_ip = raspberry_pi_ip
        self.running = False
        self.prev_frame = None
        
        # 향상된 화재 감지 시스템
        self.fire_detector = ImprovedFireDetector(model_confidence=0.4)  # 임계값 낮춤
        self.fire_alert_system = EnhancedFireAlertSystem(raspberry_pi_ip)
        
        # 통계
        self.stats = {
            'frames_received': 0,
            'fire_alerts': 0,
            'false_positives': 0,
            'start_time': None,
            'detection_methods': {'yolo': 0, 'color': 0, 'motion': 0}
        }
    
    def _draw_enhanced_detections(self, frame: np.ndarray, fire_result: dict) -> np.ndarray:
        """향상된 감지 결과 그리기"""
        for detection in fire_result['detections']:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            method = detection.get('method', 'unknown')
            
            # 감지 방법에 따른 색상 구분
            if method == 'yolo':
                color = (0, 0, 255)  # 빨간색
            elif 'color' in class_name:
                color = (0, 165, 255)  # 주황색
            else:
                color = (255, 0, 255)  # 자홍색
            
            # 바운딩 박스
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 라벨
            label = f"{class_name}: {confidence:.2f}"
            if detection.get('motion_enhanced'):
                label += " [M+]"
            
            # 텍스트 배경
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 8), 
                         (bbox[0] + text_width, bbox[1]), color, -1)
            
            # 텍스트
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 색상 마스크 오버레이
        if fire_result.get('color_mask') is not None:
            colored_mask = cv2.applyColorMap(fire_result['color_mask'], cv2.COLORMAP_HOT)
            frame = cv2.addWeighted(frame, 0.8, colored_mask, 0.2, 0)
        
        return frame
    
    def _add_enhanced_status_overlay(self, frame: np.ndarray, fire_result: dict) -> np.ndarray:
        """향상된 상태 정보 오버레이"""
        height, width = frame.shape[:2]
        
        # 상태 텍스트
        status_lines = [
            f"Frames: {self.stats['frames_received']}",
            f"Alerts: {self.stats['fire_alerts']}",
            f"Buffer: {fire_result['buffer_count']}/5",
            f"Motion: {'YES' if fire_result['motion_detected'] else 'NO'}"
        ]
        
        # 감지 방법별 통계
        yolo_count = len([d for d in fire_result['detections'] if d.get('method') == 'yolo'])
        color_count = len([d for d in fire_result['detections'] if 'color' in d.get('class_name', '')])
        
        status_lines.extend([
            f"YOLO: {yolo_count}, Color: {color_count}",
        ])
        
        # 화재 감지 상태
        if fire_result['fire_detected']:
            status_lines.insert(0, "🔥 FIRE CONFIRMED!")
            status_color = (0, 0, 255)
        elif fire_result['raw_detection']:
            status_lines.insert(0, "⚠️  Fire Detected")
            status_color = (0, 165, 255)
        else:
            status_lines.insert(0, "✅ No Fire")
            status_color = (0, 255, 0)
        
        # 배경 박스
        box_height = len(status_lines) * 20 + 10
        cv2.rectangle(frame, (10, 10), (320, box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, box_height), status_color, 2)
        
        # 텍스트
        for i, line in enumerate(status_lines):
            y_pos = 25 + i * 20
            color = status_color if i == 0 else (255, 255, 255)
            cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def receive_video(self):
        """향상된 영상 수신 및 화재 감지"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(1)
        
        print("🔥 ENHANCED Fire Detection System Started!")
        print(f"📡 Listening on port {self.port}")
        print("🎯 Multi-method detection: YOLO + Color + Motion")
        print("⚙️  Enhanced sensitivity settings active")
        print("📺 Press 'q' in video window to quit")
        print("-" * 60)
        
        try:
            self.running = True
            self.stats['start_time'] = time.time()
            
            while self.running:
                print("⏳ Waiting for Raspberry Pi connection...")
                client_socket, addr = server_socket.accept()
                
                if not self.raspberry_pi_ip:
                    self.raspberry_pi_ip = addr[0]
                    self.fire_alert_system.raspberry_pi_ip = addr[0]
                
                print(f"🔗 Connected from: {addr[0]}:{addr[1]}")
                print("🔍 Starting enhanced fire detection...")
                
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
                            
                            # 향상된 화재 감지
                            fire_result = self.fire_detector.detect_fire(frame, self.prev_frame)
                            
                            # 화재 확인 시 알림
                            if fire_result['fire_detected']:
                                if self.fire_alert_system.trigger_alert(fire_result):
                                    self.stats['fire_alerts'] += 1
                                    print(f"🚨 FIRE ALERT #{self.stats['fire_alerts']} TRIGGERED!")
                            
                            # 시각화
                            frame = self._draw_enhanced_detections(frame, fire_result)
                            frame = self._add_enhanced_status_overlay(frame, fire_result)
                            
                            cv2.imshow('🔥 Enhanced Fire Detection System', frame)
                            
                            # 이전 프레임 저장
                            self.prev_frame = frame.copy()
                            
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.running = False
                                break
                        
                except Exception as e:
                    print(f"Error during streaming: {e}")
                finally:
                    client_socket.close()
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Server error: {e}")
        finally:
            server_socket.close()
            cv2.destroyAllWindows()
            self._print_final_stats()
    
    def _print_final_stats(self):
        """최종 통계 출력"""
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            print(f"\n{'='*50}")
            print(f"🔥 ENHANCED FIRE DETECTION STATISTICS")
            print(f"{'='*50}")
            print(f"⏱️  Runtime: {runtime:.1f} seconds")
            print(f"📊 Frames processed: {self.stats['frames_received']}")
            print(f"🚨 Fire alerts: {self.stats['fire_alerts']}")
            
            if runtime > 0:
                fps = self.stats['frames_received'] / runtime
                print(f"🎯 Average FPS: {fps:.1f}")
                print(f"⚡ Alert rate: {self.stats['fire_alerts']/runtime*60:.1f} alerts/min")
            
            print(f"{'='*50}")

def main():
    """메인 함수"""
    print("🔥 ENHANCED Fire Detection System")
    print("🎯 Multi-method detection (YOLO + Color + Motion)")
    print("-" * 50)
    
    raspberry_pi_ip = None  # 자동 감지
    
    try:
        receiver = EnhancedVideoReceiver(
            port=9999, 
            raspberry_pi_ip=raspberry_pi_ip
        )
        receiver.receive_video()
        
    except Exception as e:
        print(f"❌ Failed to start system: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. pip install ultralytics torch opencv-python")
        print("2. Check Raspberry Pi video stream")
        print("3. Verify firewall settings (port 9999)")
        print("4. Consider downloading a custom fire detection model")

if __name__ == "__main__":
    main()