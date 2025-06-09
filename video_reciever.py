import socket
import cv2
import numpy as np
import struct

def receive_video():
    # 소켓 설정
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # 모든 IP에서 포트 9999로 수신 대기
    server_socket.bind(('0.0.0.0', 9999))
    server_socket.listen(1)
    
    print("Video Receiver Started!")
    print("Listening on port 9999")
    print("Waiting for Raspberry Pi connection...")
    print("Press 'q' in video window to quit")
    print("-" * 50)
    
    try:
        while True:
            # 라즈베리파이 연결 대기
            client_socket, addr = server_socket.accept()
            print(f"Connected from Raspberry Pi: {addr[0]}:{addr[1]}")
            
            try:
                while True:
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
                        # 화면에 표시
                        cv2.imshow('Raspberry Pi Video Stream', frame)
                        
                        # 'q' 키로 종료
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("User requested quit")
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
        print("Video receiver stopped")

if __name__ == "__main__":
    receive_video()