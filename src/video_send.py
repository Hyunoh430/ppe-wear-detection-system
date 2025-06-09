"""
라즈베리파이에서 영상을 다른 컴퓨터로 전송하는 모듈
가장 실용적이고 안정적인 방법들을 제공
"""

import cv2
import socket
import threading
import time
import logging
import struct
import pickle
import numpy as np
from typing import Optional, Callable
from picamera2 import Picamera2

class RPiVideoSender:
    """라즈베리파이 영상 전송기 - 가장 실용적인 방법"""
    
    def __init__(self, target_ip: str, target_port: int = 9999):
        """
        초기화
        
        Args:
            target_ip: 받을 컴퓨터의 IP 주소 (예: "192.168.1.100")
            target_port: 포트 번호 (기본값: 9999)
        """
        self.target_ip = target_ip
        self.target_port = target_port
        self.is_sending = False
        self.socket = None
        self.send_thread = None
        
        # 전송 설정 (라즈베리파이 최적화)
        self.frame_width = 320   # 작은 해상도로 빠른 전송
        self.frame_height = 240
        self.jpeg_quality = 60   # 압축률 높여서 빠른 전송
        self.fps_limit = 10      # 10fps로 제한해서 안정적 전송
        
        self.logger = logging.getLogger(__name__)
        
    def start_sending(self, camera_frame_getter: Callable[[], np.ndarray]) -> bool:
        """
        영상 전송 시작
        
        Args:
            camera_frame_getter: 카메라에서 프레임을 가져오는 함수
            
        Returns:
            bool: 성공 여부
        """
        if self.is_sending:
            self.logger.warning("Already sending video")
            return True
            
        try:
            # 소켓 연결
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)  # 5초 타임아웃
            self.socket.connect((self.target_ip, self.target_port))
            
            self.is_sending = True
            
            # 전송 스레드 시작
            self.send_thread = threading.Thread(
                target=self._send_loop,
                args=(camera_frame_getter,),
                daemon=True
            )
            self.send_thread.start()
            
            print(f"✓ Video sending started to {self.target_ip}:{self.target_port}")
            print(f"  Resolution: {self.frame_width}x{self.frame_height}")
            print(f"  Quality: {self.jpeg_quality}%, FPS: {self.fps_limit}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to start video sending: {e}")
            self.is_sending = False
            return False
    
    def stop_sending(self):
        """영상 전송 중지"""
        print("Stopping video transmission...")
        self.is_sending = False
        
        if self.send_thread:
            self.send_thread.join(timeout=3)
            
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
        print("✓ Video transmission stopped")
    
    def _send_loop(self, camera_frame_getter: Callable[[], np.ndarray]):
        """전송 메인 루프"""
        frame_interval = 1.0 / self.fps_limit
        last_send_time = 0
        
        while self.is_sending:
            try:
                current_time = time.time()
                
                # FPS 제한
                if current_time - last_send_time < frame_interval:
                    time.sleep(0.01)
                    continue
                
                # 프레임 가져오기
                frame = camera_frame_getter()
                if frame is None:
                    continue
                
                # 프레임 전송
                if self._send_frame(frame):
                    last_send_time = current_time
                else:
                    # 전송 실패 시 중지
                    break
                    
            except Exception as e:
                print(f"Send loop error: {e}")
                break
        
        self.is_sending = False
    
    def _send_frame(self, frame: np.ndarray) -> bool:
        """단일 프레임 전송"""
        try:
            # 1. 프레임 크기 조정
            resized_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            # 2. JPEG 압축
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            success, encoded_frame = cv2.imencode('.jpg', resized_frame, encode_params)
            
            if not success:
                return False
            
            # 3. 데이터 직렬화
            frame_data = encoded_frame.tobytes()
            data_size = len(frame_data)
            
            # 4. 크기 정보 전송 (4바이트)
            size_bytes = struct.pack('!I', data_size)
            self.socket.sendall(size_bytes)
            
            # 5. 실제 프레임 데이터 전송
            self.socket.sendall(frame_data)
            
            return True
            
        except Exception as e:
            print(f"Frame send error: {e}")
            return False

class SimpleWebStream:
    """간단한 웹 스트리밍 (브라우저에서 볼 수 있음)"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.is_streaming = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.server = None
        self.server_thread = None
        
    def start_streaming(self) -> bool:
        """웹 스트리밍 시작"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            
            streamer_ref = self
            
            class VideoHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        # 메인 페이지
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        
                        html = """
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>Raspberry Pi Camera</title>
                            <style>
                                body { 
                                    background: #000; 
                                    color: #fff; 
                                    text-align: center; 
                                    font-family: Arial;
                                    margin: 0;
                                    padding: 20px;
                                }
                                img { 
                                    max-width: 100%; 
                                    height: auto; 
                                    border: 2px solid #333;
                                }
                                .info {
                                    margin: 20px 0;
                                    color: #aaa;
                                }
                            </style>
                        </head>
                        <body>
                            <h1>🎥 Raspberry Pi Camera Stream</h1>
                            <div class="info">PPE Detection System - Live View</div>
                            <img src="/stream" alt="Camera Stream">
                            <div class="info">
                                Auto-refresh every frame<br>
                                Stream will pause during PPE detection
                            </div>
                        </body>
                        </html>
                        """
                        self.wfile.write(html.encode())
                        
                    elif self.path == '/stream':
                        # 실제 스트림
                        self.send_response(200)
                        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
                        self.end_headers()
                        
                        while streamer_ref.is_streaming:
                            with streamer_ref.frame_lock:
                                if streamer_ref.latest_frame is not None:
                                    # 프레임을 JPEG로 인코딩
                                    success, buffer = cv2.imencode('.jpg', streamer_ref.latest_frame, 
                                                                 [cv2.IMWRITE_JPEG_QUALITY, 70])
                                    if success:
                                        frame_bytes = buffer.tobytes()
                                        
                                        # 멀티파트 응답 전송
                                        self.wfile.write(b'\r\n--frame\r\n')
                                        self.send_header('Content-type', 'image/jpeg')
                                        self.send_header('Content-length', len(frame_bytes))
                                        self.end_headers()
                                        self.wfile.write(frame_bytes)
                            
                            time.sleep(0.1)  # 10 FPS
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def log_message(self, format, *args):
                    pass  # 로그 억제
            
            # HTTP 서버 시작
            self.server = HTTPServer(('0.0.0.0', self.port), VideoHandler)
            self.is_streaming = True
            
            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True
            )
            self.server_thread.start()
            
            print(f"✓ Web streaming started!")
            print(f"  Open in browser: http://라즈베리파이IP:{self.port}")
            print(f"  Local access: http://localhost:{self.port}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to start web streaming: {e}")
            return False
    
    def update_frame(self, frame: np.ndarray):
        """프레임 업데이트"""
        with self.frame_lock:
            # 적당한 크기로 조정
            resized = cv2.resize(frame, (640, 480))
            self.latest_frame = resized
    
    def stop_streaming(self):
        """웹 스트리밍 중지"""
        self.is_streaming = False
        
        if self.server:
            self.server.shutdown()
            
        if self.server_thread:
            self.server_thread.join(timeout=2)
            
        print("✓ Web streaming stopped")

# =======================================================
# 받는 쪽 프로그램 (다른 컴퓨터에서 실행)
# =======================================================

def create_receiver_program():
    """받는 쪽 프로그램 코드 생성"""
    receiver_code = '''
"""
영상 수신 프로그램 - 다른 컴퓨터에서 실행
라즈베리파이에서 보내는 영상을 받아서 화면에 표시
"""

import cv2
import socket
import struct
import numpy as np

def receive_video(port=9999):
    """영상 수신"""
    print("Waiting for video stream...")
    print(f"Listening on port {port}")
    print("Press 'q' to quit")
    
    # 서버 소켓 생성
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(1)
    
    try:
        # 연결 대기
        print("Waiting for connection...")
        client_socket, address = server_socket.accept()
        print(f"✓ Connected from {address}")
        
        while True:
            try:
                # 1. 데이터 크기 수신 (4바이트)
                size_data = client_socket.recv(4)
                if len(size_data) < 4:
                    break
                    
                data_size = struct.unpack('!I', size_data)[0]
                
                # 2. 실제 프레임 데이터 수신
                frame_data = b''
                while len(frame_data) < data_size:
                    packet = client_socket.recv(data_size - len(frame_data))
                    if not packet:
                        break
                    frame_data += packet
                
                if len(frame_data) != data_size:
                    break
                
                # 3. JPEG 디코딩
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # 4. 화면에 표시
                    cv2.imshow('Raspberry Pi Camera', frame)
                    
                    # 'q' 키로 종료
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
            except Exception as e:
                print(f"Receive error: {e}")
                break
                
    except KeyboardInterrupt:
        print("\\nInterrupted by user")
    finally:
        client_socket.close()
        server_socket.close()
        cv2.destroyAllWindows()
        print("✓ Video receiver stopped")

if __name__ == "__main__":
    receive_video()
'''
    
    # 파일 저장
    with open('video_receiver.py', 'w', encoding='utf-8') as f:
        f.write(receiver_code)
    
    print("✓ video_receiver.py created!")
    print("  Copy this file to your computer and run it to receive video")

# =======================================================
# 테스트 및 사용법
# =======================================================

def test_video_sender():
    """영상 전송 테스트"""
    print("=" * 50)
    print("라즈베리파이 영상 전송 테스트")
    print("=" * 50)
    
    # 1. 수신자 IP 입력
    target_ip = input("받을 컴퓨터의 IP 주소를 입력하세요 (예: 192.168.1.100): ").strip()
    if not target_ip:
        print("IP 주소가 필요합니다!")
        return False
    
    try:
        # 2. 카메라 초기화
        print("카메라 초기화 중...")
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        time.sleep(2)
        print("✓ 카메라 준비 완료")
        
        # 3. 영상 전송기 초기화
        sender = RPiVideoSender(target_ip)
        
        # 4. 전송 시작
        print(f"\\n{target_ip}로 영상 전송 시작...")
        print("받는 컴퓨터에서 video_receiver.py를 먼저 실행해주세요!")
        input("준비되면 Enter를 눌러주세요...")
        
        if sender.start_sending(lambda: camera.capture_array()):
            print("\\n전송 중... (Ctrl+C로 중지)")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\\n사용자가 중지했습니다")
        
        # 5. 정리
        sender.stop_sending()
        camera.stop()
        print("✓ 테스트 완료")
        return True
        
    except Exception as e:
        print(f"✗ 테스트 실패: {e}")
        return False

def test_web_streaming():
    """웹 스트리밍 테스트"""
    print("=" * 50)
    print("웹 스트리밍 테스트")
    print("=" * 50)
    
    try:
        # 카메라 초기화
        print("카메라 초기화 중...")
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        time.sleep(2)
        
        # 웹 스트리밍 시작
        web_streamer = SimpleWebStream(8080)
        
        if web_streamer.start_streaming():
            print("\\n웹 스트리밍 시작됨!")
            print("브라우저에서 접속하세요:")
            print("  http://라즈베리파이IP:8080")
            print("\\n60초 동안 스트리밍... (Ctrl+C로 중지)")
            
            try:
                for i in range(600):  # 60초
                    frame = camera.capture_array()
                    web_streamer.update_frame(frame)
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\\n사용자가 중지했습니다")
        
        web_streamer.stop_streaming()
        camera.stop()
        print("✓ 웹 스트리밍 테스트 완료")
        return True
        
    except Exception as e:
        print(f"✗ 웹 스트리밍 실패: {e}")
        return False

if __name__ == "__main__":
    print("🎥 라즈베리파이 영상 전송 프로그램")
    print("=" * 50)
    print("1. 다른 컴퓨터로 직접 전송 (추천)")
    print("2. 웹 브라우저로 스트리밍")
    print("3. 받는 프로그램 생성 (video_receiver.py)")
    print("4. 모든 테스트")
    
    choice = input("\\n선택하세요 (1-4): ").strip()
    
    if choice == "1":
        test_video_sender()
    elif choice == "2":
        test_web_streaming()
    elif choice == "3":
        create_receiver_program()
    elif choice == "4":
        create_receiver_program()
        print("\\n" + "="*30)
        test_web_streaming()
        print("\\n" + "="*30)
        test_video_sender()
    else:
        print("잘못된 선택입니다")