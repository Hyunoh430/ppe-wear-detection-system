"""
Servo motor controller for waste disposal door with position holding
"""

import RPi.GPIO as GPIO
import time
import threading
import logging
from typing import Optional
from enum import Enum

from config import *

class DoorState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    MOVING = "moving"
    ERROR = "error"

class ServoController:
    def __init__(self, pin: int = SERVO_PIN):
        """Initialize servo motor controller"""
        self.logger = logging.getLogger(__name__)
        self.pin = pin
        self.servo: Optional[GPIO.PWM] = None
        self.current_angle = SERVO_CLOSED_ANGLE
        self.target_angle = SERVO_CLOSED_ANGLE
        self.state = DoorState.CLOSED
        self.is_initialized = False
        self.movement_lock = threading.Lock()
        
        # Position holding related
        self.hold_position = False  # 위치 유지 모드
        self.hold_thread: Optional[threading.Thread] = None
        self.hold_stop_event = threading.Event()
        
        self._initialize_gpio()
    
    def _initialize_gpio(self):
        """Initialize GPIO and PWM for servo"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            
            self.servo = GPIO.PWM(self.pin, SERVO_FREQUENCY)
            self.servo.start(0)
            
            # Set initial position (closed) and start holding
            self._set_angle_and_hold(SERVO_CLOSED_ANGLE)
            
            self.is_initialized = True
            self.logger.info(f"Servo controller initialized on GPIO pin {self.pin}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize servo controller: {e}")
            self.state = DoorState.ERROR
            raise
    
    def _calculate_duty_cycle(self, angle: float) -> float:
        """Calculate PWM duty cycle for given angle"""
        if angle < 0:
            angle = 0
        elif angle > 180:
            angle = 180
        
        duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
        return duty
    
    def _hold_position_worker(self):
        """Background thread to maintain servo position"""
        while not self.hold_stop_event.is_set():
            if self.hold_position and self.servo:
                try:
                    duty = self._calculate_duty_cycle(self.current_angle)
                    self.servo.ChangeDutyCycle(duty)
                    # 짧은 간격으로 PWM 신호를 계속 보냄
                    time.sleep(0.02)  # 20ms마다 신호 갱신 (50Hz)
                except Exception as e:
                    self.logger.error(f"Position holding error: {e}")
                    break
            else:
                time.sleep(0.1)  # 대기 모드에서는 더 긴 간격
    
    def _start_position_holding(self):
        """Start position holding thread"""
        if self.hold_thread is None or not self.hold_thread.is_alive():
            self.hold_stop_event.clear()
            self.hold_thread = threading.Thread(target=self._hold_position_worker, daemon=True)
            self.hold_thread.start()
            self.logger.debug("Position holding thread started")
    
    def _stop_position_holding(self):
        """Stop position holding thread"""
        self.hold_position = False
        if self.servo:
            self.servo.ChangeDutyCycle(0)  # PWM 신호 정지
        
        if self.hold_thread and self.hold_thread.is_alive():
            self.hold_stop_event.set()
            self.hold_thread.join(timeout=1)
        self.logger.debug("Position holding stopped")
    
    def _set_angle_and_hold(self, angle: float, hold_time: float = 0.5):
        """Set servo to specific angle and maintain position"""
        if not self.is_initialized or self.servo is None:
            self.logger.error("Servo not initialized")
            return False
        
        try:
            # 이동 중에는 위치 유지 중단
            self.hold_position = False
            time.sleep(0.1)  # 기존 PWM 신호가 안정화될 시간
            
            duty = self._calculate_duty_cycle(angle)
            
            # 목표 위치로 이동
            self.servo.ChangeDutyCycle(duty)
            time.sleep(hold_time)  # 이동 완료까지 대기
            
            # 새로운 위치 저장
            self.current_angle = angle
            
            # 위치 유지 시작 (열린 상태에서만)
            if angle == SERVO_OPEN_ANGLE:
                self.hold_position = True
                self._start_position_holding()
                self.logger.debug(f"Servo moved to {angle}° with position holding enabled")
            else:
                # 닫힌 상태에서는 위치 유지 안함 (전력 절약)
                self.servo.ChangeDutyCycle(0)
                self.logger.debug(f"Servo moved to {angle}° without position holding")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set servo angle: {e}")
            self.state = DoorState.ERROR
            return False
    
    def _move_to_angle_smooth(self, target_angle: float, speed: int = 2):
        """Move servo smoothly to target angle with position holding for open state"""
        if not self.is_initialized:
            return False
        
        self.state = DoorState.MOVING
        current = int(self.current_angle)
        target = int(target_angle)
        
        if current == target:
            self.logger.debug(f"Already at target angle {target}°")
            return True
        
        try:
            # 이동 중에는 위치 유지 중단
            self.hold_position = False
            time.sleep(0.1)
            
            # 문 여는 경우 (100도 → 70도, 빠르게)
            if current == SERVO_CLOSED_ANGLE and target == SERVO_OPEN_ANGLE:
                self.logger.debug("Opening door: fast movement (100° → 70°)")
                for deg in range(current, target - 1, -1):
                    duty = self._calculate_duty_cycle(deg)
                    self.servo.ChangeDutyCycle(duty)
                    time.sleep(SERVO_MOVE_DELAY_FAST)  # 빠른 속도 (0.005초)
            
            # 문 닫는 경우 (70도 → 100도, 천천히)
            elif current == SERVO_OPEN_ANGLE and target == SERVO_CLOSED_ANGLE:
                self.logger.debug("Closing door: slow movement (70° → 100°)")
                for deg in range(current, target + 1):
                    duty = self._calculate_duty_cycle(deg)
                    self.servo.ChangeDutyCycle(duty)
                    time.sleep(SERVO_MOVE_DELAY_SLOW)  # 느린 속도 (0.03초)
            
            # 기타 경우 (기존 로직 유지)
            else:
                self.logger.debug(f"General movement: {current}° → {target}°")
                if current < target:
                    step = speed
                    angle_range = range(current, target + 1, step)
                else:
                    step = -speed
                    angle_range = range(current, target - 1, step)
                
                # Move through intermediate angles quickly
                for angle in angle_range[:-1]:  # Exclude last angle
                    duty = self._calculate_duty_cycle(angle)
                    self.servo.ChangeDutyCycle(duty)
                    time.sleep(SERVO_MOVE_DELAY_FAST)
            
            # 최종 위치로 이동 및 위치 유지 설정
            self._set_angle_and_hold(target_angle, hold_time=0.8)
            self.target_angle = target_angle
            
            return True
            
        except Exception as e:
            self.logger.error(f"Smooth movement failed: {e}")
            self.state = DoorState.ERROR
            return False
    
    def open_door(self, smooth: bool = True) -> bool:
        """Open the waste disposal door and maintain position"""
        with self.movement_lock:
            if self.state == DoorState.ERROR:
                self.logger.error("Cannot open door - servo in error state")
                return False
            
            if self.state == DoorState.OPEN:
                self.logger.info("Door already open")
                return True
            
            self.logger.info("Opening waste disposal door...")
            
            if smooth:
                success = self._move_to_angle_smooth(SERVO_OPEN_ANGLE)
            else:
                success = self._set_angle_and_hold(SERVO_OPEN_ANGLE)
            
            if success:
                self.state = DoorState.OPEN
                self.logger.info(f"Door opened successfully (angle: {self.current_angle}°) - position holding active")
            else:
                self.logger.error("Failed to open door")
                self.state = DoorState.ERROR
            
            return success
    
    def close_door(self, smooth: bool = True) -> bool:
        """Close the waste disposal door and stop position holding"""
        with self.movement_lock:
            if self.state == DoorState.ERROR:
                self.logger.error("Cannot close door - servo in error state")
                return False
            
            if self.state == DoorState.CLOSED:
                self.logger.info("Door already closed")
                return True
            
            self.logger.info("Closing waste disposal door...")
            
            # 위치 유지 중단
            self._stop_position_holding()
            
            if smooth:
                success = self._move_to_angle_smooth(SERVO_CLOSED_ANGLE)
            else:
                success = self._set_angle_and_hold(SERVO_CLOSED_ANGLE)
            
            if success:
                self.state = DoorState.CLOSED
                self.logger.info(f"Door closed successfully (angle: {self.current_angle}°) - position holding disabled")
            else:
                self.logger.error("Failed to close door")
                self.state = DoorState.ERROR
            
            return success
    
    def emergency_stop(self):
        """Emergency stop - immediately stop servo movement and position holding"""
        try:
            self._stop_position_holding()
            self.state = DoorState.ERROR
            self.logger.warning("Emergency stop activated")
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
    
    def get_door_state(self) -> DoorState:
        """Get current door state"""
        return self.state
    
    def get_current_angle(self) -> float:
        """Get current servo angle"""
        return self.current_angle
    
    def is_door_open(self) -> bool:
        """Check if door is open"""
        return self.state == DoorState.OPEN
    
    def is_door_closed(self) -> bool:
        """Check if door is closed"""
        return self.state == DoorState.CLOSED
    
    def is_holding_position(self) -> bool:
        """Check if position holding is active"""
        return self.hold_position
    
    def test_movement(self) -> bool:
        """Test servo movement - open and close cycle"""
        self.logger.info("Testing servo movement with position holding...")
        
        try:
            # Test opening
            if not self.open_door():
                return False
            
            self.logger.info(f"Door open - position holding: {self.is_holding_position()}")
            time.sleep(5)  # 더 긴 테스트 시간
            
            # Test closing
            if not self.close_door():
                return False
            
            self.logger.info(f"Door closed - position holding: {self.is_holding_position()}")
            self.logger.info("Servo test completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Servo test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up GPIO resources"""
        try:
            # 위치 유지 중단
            self._stop_position_holding()
            
            if self.servo:
                self.servo.stop()
            GPIO.cleanup()
            self.is_initialized = False
            self.logger.info("Servo controller cleaned up")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


# ==========================================
# 개별 테스트 코드 (수정된 버전)
# ==========================================

def test_servo_position_holding():
    """Position holding test"""
    print("=" * 50)
    print("Servo Motor Position Holding Test")
    print("=" * 50)
    
    try:
        print("1. Servo controller initialization...")
        with ServoController() as servo:
            print("   O Initialization successful!")
            print(f"   Current angle: {servo.get_current_angle()}°")
            print(f"   Door state: {servo.get_door_state().value}")
            print(f"   Position holding: {servo.is_holding_position()}")
            
            print("\n2. Door opening test...")
            if servo.open_door():
                print("   O Door opening successful!")
                print(f"   Current angle: {servo.get_current_angle()}°")
                print(f"   Door state: {servo.get_door_state().value}")
                print(f"   Position holding: {servo.is_holding_position()}")
            else:
                print("   X Door opening failed!")
                return False
            
            print("\n3. Position holding test (10 seconds)...")
            print("   Try applying external force to the door!")
            for i in range(10):
                time.sleep(1)
                print(f"   {i+1}/10 seconds - Holding: {servo.is_holding_position()}")
            
            print("\n4. Door closing test...")
            if servo.close_door():
                print("   O Door closing successful!")
                print(f"   Current angle: {servo.get_current_angle()}°")
                print(f"   Door state: {servo.get_door_state().value}")
                print(f"   Position holding: {servo.is_holding_position()}")
            else:
                print("   X Door closing failed!")
                return False
            
            print("\n5. Test completed!")
            print("   O Position holding servo motor working normally!")
            
        return True
        
    except Exception as e:
        print(f"   X Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_servo_extended_open():
    """Extended open test with position holding"""
    print("=" * 50)
    print("Extended Door Open Test (30 seconds)")
    print("=" * 50)
    print("This test will keep the door open for 30 seconds")
    print("Try applying pressure to test position holding!")
    
    try:
        with ServoController() as servo:
            print("\n1. Opening door...")
            if not servo.open_door():
                print("X Failed to open door")
                return False
            
            print("O Door opened with position holding active")
            print(f"Position holding status: {servo.is_holding_position()}")
            
            print("\n2. Maintaining open position for 30 seconds...")
            for i in range(30):
                time.sleep(1)
                if i % 5 == 0:  # 5초마다 상태 출력
                    print(f"   {i+1}/30s - Angle: {servo.get_current_angle()}° - Holding: {servo.is_holding_position()}")
            
            print("\n3. Closing door...")
            if servo.close_door():
                print("O Door closed successfully")
                print(f"Position holding status: {servo.is_holding_position()}")
            else:
                print("X Failed to close door")
                return False
            
            print("\nO Extended open test completed!")
            
        return True
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        return True
    except Exception as e:
        print(f"X Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    print("Servo Motor Controller Test Options (Position Holding Version):")
    print("1. Basic test (open/close)")
    print("2. Position holding test")
    print("3. Extended open test (30s)")
    print("4. Manual control test")
    
    choice = input("Select (1-4): ").strip()
    
    if choice == "1":
        success = test_servo_basic()
    elif choice == "2":
        success = test_servo_position_holding()
    elif choice == "3":
        success = test_servo_extended_open()
    elif choice == "4":
        success = test_servo_manual_control()
    else:
        print("Invalid selection. Running position holding test.")
        success = test_servo_position_holding()
    
    if success:
        print("\n Test successful!")
    else:
        print("\n Test failed!")
        sys.exit(1)