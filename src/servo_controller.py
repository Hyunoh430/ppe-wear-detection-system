"""
Servo motor controller for waste disposal door - Simplified version
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
        """Calculate PWM duty cycle for given angle"""
        if angle < 0:
            angle = 0
        elif angle > 180:
            angle = 180
        
        duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
        return duty
    
    def _move_to_angle_direct(self, angle: float):
        """Move servo directly to angle (simple version)"""
        if not self.is_initialized or self.servo is None:
            return False
        
        try:
            duty = self._calculate_duty_cycle(angle)
            self.servo.ChangeDutyCycle(duty)
            time.sleep(0.5)  # 이동 완료 대기
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
            
            try:
                if smooth:
                    # 부드러운 이동: 100도 → 70도 (빠르게)
                    for deg in range(int(SERVO_CLOSED_ANGLE), int(SERVO_OPEN_ANGLE) - 1, -1):
                        duty = self._calculate_duty_cycle(deg)
                        self.servo.ChangeDutyCycle(duty)
                        time.sleep(0.005)  # 빠른 속도
                else:
                    # 직접 이동
                    self._move_to_angle_direct(SERVO_OPEN_ANGLE)
                
                # 최종 위치에서 계속 힘을 가함 (홀드)
                final_duty = self._calculate_duty_cycle(SERVO_OPEN_ANGLE)
                self.servo.ChangeDutyCycle(final_duty)
                
                self.current_angle = SERVO_OPEN_ANGLE
                self.state = DoorState.OPEN
                
                self.logger.info(f"Door opened successfully (angle: {self.current_angle}°) - holding position")
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
            
            try:
                if smooth:
                    # 부드러운 이동: 70도 → 100도 (천천히)
                    for deg in range(int(SERVO_OPEN_ANGLE), int(SERVO_CLOSED_ANGLE) + 1):
                        duty = self._calculate_duty_cycle(deg)
                        self.servo.ChangeDutyCycle(duty)
                        time.sleep(0.03)  # 느린 속도
                else:
                    # 직접 이동
                    self._move_to_angle_direct(SERVO_CLOSED_ANGLE)
                
                # 닫힌 후에는 모터 정지
                time.sleep(0.5)  # 완전 닫힘 대기
                self.servo.ChangeDutyCycle(0)  # PWM 정지
                
                self.current_angle = SERVO_CLOSED_ANGLE
                self.state = DoorState.CLOSED
                
                self.logger.info(f"Door closed successfully (angle: {self.current_angle}°) - motor stopped")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to close door: {e}")
                self.state = DoorState.ERROR
                return False
    
    def emergency_stop(self):
        """Emergency stop - immediately stop servo movement"""
        try:
            if self.servo:
                self.servo.ChangeDutyCycle(0)  # Stop PWM signal
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
    
    def test_movement(self) -> bool:
        """Test servo movement - 당신의 테스트 코드와 동일한 방식"""
        self.logger.info("Testing servo movement...")
        
        try:
            # Open: 100 → 70 (fast)
            self.logger.info("Opening door (100° → 70°)...")
            for deg in range(100, 69, -1):
                duty = self._calculate_duty_cycle(deg)
                self.servo.ChangeDutyCycle(duty)
                time.sleep(0.005)
            
            # Hold at 70 degrees for 3 seconds
            self.logger.info("Holding at 70° for 3 seconds...")
            hold_duty = self._calculate_duty_cycle(70)
            self.servo.ChangeDutyCycle(hold_duty)
            time.sleep(3.0)
            
            # Close: 70 → 100 (slow)
            self.logger.info("Closing door (70° → 100°)...")
            for deg in range(70, 101):
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
            if self.servo:
                # Stop PWM before cleanup
                self.servo.ChangeDutyCycle(0)
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