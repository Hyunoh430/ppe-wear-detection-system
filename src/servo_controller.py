"""
Servo motor controller - Fixed for main.py integration
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
        
        # PWM 유지를 위한 추가 변수들
        self.hold_thread: Optional[threading.Thread] = None
        self.should_hold = False
        self.hold_angle = None
        
        self._initialize_gpio()
    
    def _initialize_gpio(self):
        """Initialize GPIO and PWM for servo"""
        try:
            GPIO.cleanup()  # 먼저 클린업
            time.sleep(0.2)
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            
            self.servo = GPIO.PWM(self.pin, SERVO_FREQUENCY)
            self.servo.start(0)
            
            # 초기 위치로 이동 (닫힌 상태)
            self._move_to_angle_direct(SERVO_CLOSED_ANGLE)
            time.sleep(1)
            self.servo.ChangeDutyCycle(0)  # 초기에는 정지
            
            self.is_initialized = True
            self.logger.info(f"Servo controller initialized on GPIO pin {self.pin}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize servo controller: {e}")
            self.state = DoorState.ERROR
            raise
    
    def _calculate_duty_cycle(self, angle: float) -> float:
        """Calculate PWM duty cycle for given angle - 하드코딩으로 안정성 확보"""
        if angle < 0:
            angle = 0
        elif angle > 180:
            angle = 180
        
        # 테스트 코드와 동일한 값 사용
        SERVO_MIN_DUTY = 3
        SERVO_MAX_DUTY = 12
        duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
        return duty
    
    def _hold_position_thread(self):
        """별도 스레드에서 위치 유지 (main.py 간섭 방지)"""
        self.logger.info(f"Starting position hold thread for {self.hold_angle}°")
        
        while self.should_hold and self.is_initialized:
            try:
                if self.servo and self.hold_angle is not None:
                    duty = self._calculate_duty_cycle(self.hold_angle)
                    self.servo.ChangeDutyCycle(duty)
                time.sleep(0.1)  # 0.1초마다 PWM 신호 재전송
            except Exception as e:
                self.logger.error(f"Hold position thread error: {e}")
                break
        
        self.logger.info("Position hold thread stopped")
    
    def _start_holding_position(self, angle: float):
        """위치 유지 시작"""
        self._stop_holding_position()  # 기존 홀드 중지
        
        self.hold_angle = angle
        self.should_hold = True
        self.hold_thread = threading.Thread(target=self._hold_position_thread, daemon=True)
        self.hold_thread.start()
    
    def _stop_holding_position(self):
        """위치 유지 중지"""
        self.should_hold = False
        if self.hold_thread and self.hold_thread.is_alive():
            self.hold_thread.join(timeout=1.0)
        self.hold_thread = None
        self.hold_angle = None
    
    def _move_to_angle_direct(self, angle: float):
        """Move servo directly to angle"""
        if not self.is_initialized or self.servo is None:
            return False
        
        try:
            duty = self._calculate_duty_cycle(angle)
            self.servo.ChangeDutyCycle(duty)
            time.sleep(0.8)
            self.current_angle = angle
            return True
        except Exception as e:
            self.logger.error(f"Direct movement failed: {e}")
            return False
    
    def open_door(self, smooth: bool = True) -> bool:
        """Open the waste disposal door"""
        with self.movement_lock:
            if self.state == DoorState.ERROR:
                self.logger.error("Cannot open door - servo in error state")
                return False
            
            if self.state == DoorState.OPEN:
                self.logger.info("Door already open")
                return True
            
            self.logger.info("Opening waste disposal door...")
            self.state = DoorState.MOVING
            
            # 기존 홀드 중지
            self._stop_holding_position()
            
            try:
                if smooth:
                    # 부드러운 이동: 100도 → 70도 (빠르게)
                    self.logger.info("Moving from 100° to 65°...")
                    for deg in range(100, 64, -1):
                        duty = self._calculate_duty_cycle(deg)
                        self.servo.ChangeDutyCycle(duty)
                        time.sleep(0.005)
                    self.logger.info("Movement completed")
                else:
                    self._move_to_angle_direct(70)
                
                # 위치 유지 스레드 시작
                self.current_angle = 65
                self.state = DoorState.OPEN
                self._start_holding_position(65)
                
                self.logger.info("Door opened successfully - holding position with dedicated thread")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to open door: {e}")
                self.state = DoorState.ERROR
                return False
    
    def close_door(self, smooth: bool = True) -> bool:
        """Close the waste disposal door"""
        with self.movement_lock:
            if self.state == DoorState.ERROR:
                self.logger.error("Cannot close door - servo in error state")
                return False
            
            if self.state == DoorState.CLOSED:
                self.logger.info("Door already closed")
                return True
            
            self.logger.info("Closing waste disposal door...")
            self.state = DoorState.MOVING
            
            # 홀드 중지
            self._stop_holding_position()
            
            try:
                if smooth:
                    # 부드러운 이동: 70도 → 100도 (천천히)
                    self.logger.info("Moving from 70° to 100°...")
                    for deg in range(65, 101):
                        duty = self._calculate_duty_cycle(deg)
                        self.servo.ChangeDutyCycle(duty)
                        time.sleep(0.03)
                    self.logger.info("Movement completed")
                else:
                    self._move_to_angle_direct(100)
                
                # 닫힌 후에는 모터 정지
                time.sleep(0.5)
                self.servo.ChangeDutyCycle(0)
                self.logger.info("Motor stopped")
                
                self.current_angle = 100
                self.state = DoorState.CLOSED
                
                self.logger.info("Door closed successfully - motor stopped")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to close door: {e}")
                self.state = DoorState.ERROR
                return False
    
    def emergency_stop(self):
        """Emergency stop"""
        try:
            self._stop_holding_position()  # 홀드 스레드 중지
            if self.servo:
                self.servo.ChangeDutyCycle(0)
            self.state = DoorState.ERROR
            self.logger.warning("Emergency stop activated")
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
    
    def get_door_state(self) -> DoorState:
        return self.state
    
    def get_current_angle(self) -> float:
        return self.current_angle
    
    def is_door_open(self) -> bool:
        return self.state == DoorState.OPEN
    
    def is_door_closed(self) -> bool:
        return self.state == DoorState.CLOSED
    
    def test_movement(self) -> bool:
        """Test servo movement"""
        self.logger.info("Testing servo movement...")
        
        try:
            # 기존 홀드 중지
            self._stop_holding_position()
            
            # Open: 100 → 70 (fast)
            self.logger.info("Opening door (100° → 70°)...")
            for deg in range(100, 64, -1):
                duty = self._calculate_duty_cycle(deg)
                self.servo.ChangeDutyCycle(duty)
                time.sleep(0.005)
            
            # Hold at 70 degrees for 3 seconds with thread
            self.logger.info("Holding at 70° for 3 seconds...")
            self._start_holding_position(65)
            time.sleep(3.0)
            self._stop_holding_position()
            
            # Close: 70 → 100 (slow)
            self.logger.info("Closing door (70° → 100°)...")
            for deg in range(65, 101):
                duty = self._calculate_duty_cycle(deg)
                self.servo.ChangeDutyCycle(duty)
                time.sleep(0.03)
            
            # Stop motor after closing
            time.sleep(0.5)
            self.servo.ChangeDutyCycle(0)
            
            self.logger.info("Servo test completed successfully")
            self.current_angle = 100
            self.state = DoorState.CLOSED
            return True
            
        except Exception as e:
            self.logger.error(f"Servo test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up GPIO resources"""
        try:
            # 홀드 스레드 중지
            self._stop_holding_position()
            
            if self.servo:
                self.servo.ChangeDutyCycle(0)
                self.servo.stop()
            GPIO.cleanup()
            self.is_initialized = False
            self.logger.info("Servo controller cleaned up")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# ==========================================
# 간단한 테스트 코드
# ==========================================

def test_servo_simple():
    """Simple servo motor test - 당신의 코드 스타일로"""
    print("=" * 50)
    print("Simple Servo Motor Test")
    print("=" * 50)
    
    try:
        with ServoController() as servo:
            print("1. Testing door open...")
            if servo.open_door():
                print("   O Door opened! (Motor holding position)")
                time.sleep(3)  # 3초간 열린 상태 유지
            
            print("2. Testing door close...")
            if servo.close_door():
                print("   O Door closed! (Motor stopped)")
            
            print("3. Running full test cycle...")
            if servo.test_movement():
                print("   O Full test successful!")
            
        return True
        
    except Exception as e:
        print(f"X Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Simplified Servo Controller Test")
    success = test_servo_simple()
    
    if success:
        print("\n✅ Test successful!")
    else:
        print("\n❌ Test failed!")
        import sys
        sys.exit(1)