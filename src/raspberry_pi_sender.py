import cv2
import socket
import struct
import time
from picamera2 import Picamera2

def send_video_to_computer():
    # 컴퓨터 IP 주소 (올바른 IP로 수정)
    COMPUTER_IP = "172.20.10.4"  # 컴퓨터 실제 IP
    PORT = 9999
    
    print("Starting video sender...")
    print(f"Target computer: {COMPUTER_IP}:{PORT}")
    
    # Picamera2 설정
    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    
    # 카메라 안정화 대기
    time.sleep(2)
    
    print("Camera initialized")
    
    # 소켓 연결
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((COMPUTER_IP, PORT))
        print(f"Connected to {COMPUTER_IP}:{PORT}")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Picamera2에서 프레임 캡처 (RGB 형식)
            frame = camera.capture_array()
            
            # RGB 그대로 JPEG 인코딩
            # OpenCV는 BGR을 기대하지만, RGB로도 인코딩 가능
            
            # JPEG 압축 (품질 80%)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            data = buffer.tobytes()
            
            # 데이터 크기를 먼저 전송 (4바이트, big-endian)
            size = len(data)
            sock.sendall(struct.pack('!I', size))
            
            # 실제 프레임 데이터 전송
            sock.sendall(data)
            
            frame_count += 1
            
            # FPS 제한 (약 20 FPS)
            time.sleep(0.05)
            
            # 5초마다 통계 출력
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Frames sent: {frame_count}, FPS: {fps:.1f}")
                
    except KeyboardInterrupt:
        print("\nStopping video stream...")
    except ConnectionRefusedError:
        print("Error: Cannot connect to computer")
        print("Make sure the receiver is running on the computer")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.stop()
        sock.close()
        print("Camera released and connection closed")

if __name__ == "__main__":
    send_video_to_computer()