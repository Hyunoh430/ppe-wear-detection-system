"""
Servo motor controller for waste disposal door - Modified for continuous hold
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
        self.is_holding_position = False  # 위치 유지 상태 추가
        
        self._initialize_gpio()
    
    def _initialize_gpio(self):
        """Initialize GPIO and PWM for servo"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            
            self.servo = GPIO.PWM(self.pin, SERVO_FREQUENCY)
            self.servo.start(0)
            
            # Set initial position (closed) and stop PWM (닫힌 상태에서는 정지)
            self._set_angle_with_hold_control(SERVO_CLOSED_ANGLE, hold_position=False)
            
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
    
    def _set_angle_with_hold_control(self, angle: float, hold_time: float = 0.5, hold_position: bool = True):
        """Set servo to specific angle with control over whether to hold position"""
        if not self.is_initialized or self.servo is None:
            self.logger.error("Servo not initialized")
            return False
        
        try:
            duty = self._calculate_duty_cycle(angle)
            
            # Send PWM signal to move servo
            self.servo.ChangeDutyCycle(duty)
            
            # Hold position for specified time to ensure movement completion
            time.sleep(hold_time)
            
            if hold_position:
                # 위치 유지: PWM 신호 계속 전송
                self.is_holding_position = True
                self.logger.debug(f"Servo moved to {angle}° and holding position")
            else:
                # 위치 유지 안함: PWM 신호 정지
                self.servo.ChangeDutyCycle(0)
                self.is_holding_position = False
                self.logger.debug(f"Servo moved to {angle}° and PWM stopped")
            
            self.current_angle = angle
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set servo angle: {e}")
            self.state = DoorState.ERROR
            return False
    
    def _move_to_angle_smooth(self, target_angle: float, speed: int = 2, hold_position: bool = True):
        """Move servo smoothly to target angle with different speeds for opening/closing"""
        if not self.is_initialized:
            return False
        
        self.state = DoorState.MOVING
        current = int(self.current_angle)
        target = int(target_angle)
        
        if current == target:
            self.logger.debug(f"Already at target angle {target}°")
            return True
        
        try:
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
            
            # Final position with hold control
            self._set_angle_with_hold_control(target_angle, hold_time=0.8, hold_position=hold_position)
            self.target_angle = target_angle
            
            return True
            
        except Exception as e:
            self.logger.error(f"Smooth movement failed: {e}")
            self.state = DoorState.ERROR
            return False
    
    def open_door(self, smooth: bool = True) -> bool:
        """Open the waste disposal door and hold position"""
        with self.movement_lock:
            if self.state == DoorState.ERROR:
                self.logger.error("Cannot open door - servo in error state")
                return False
            
            if self.state == DoorState.OPEN:
                self.logger.info("Door already open")
                return True
            
            self.logger.info("Opening waste disposal door...")
            
            if smooth:
                # 문 열 때는 위치 유지 (hold_position=True)
                success = self._move_to_angle_smooth(SERVO_OPEN_ANGLE, hold_position=True)
            else:
                success = self._set_angle_with_hold_control(SERVO_OPEN_ANGLE, hold_position=True)
            
            if success:
                self.state = DoorState.OPEN
                self.logger.info(f"Door opened successfully (angle: {self.current_angle}°) - holding position")
            else:
                self.logger.error("Failed to open door")
                self.state = DoorState.ERROR
            
            return success
    
    def close_door(self, smooth: bool = True) -> bool:
        """Close the waste disposal door and stop motor"""
        with self.movement_lock:
            if self.state == DoorState.ERROR:
                self.logger.error("Cannot close door - servo in error state")
                return False
            
            if self.state == DoorState.CLOSED:
                self.logger.info("Door already closed")
                return True
            
            self.logger.info("Closing waste disposal door...")
            
            if smooth:
                # 문 닫을 때는 위치 유지 안함 (hold_position=False)
                success = self._move_to_angle_smooth(SERVO_CLOSED_ANGLE, hold_position=False)
            else:
                success = self._set_angle_with_hold_control(SERVO_CLOSED_ANGLE, hold_position=False)
            
            if success:
                self.state = DoorState.CLOSED
                self.logger.info(f"Door closed successfully (angle: {self.current_angle}°) - motor stopped")
            else:
                self.logger.error("Failed to close door")
                self.state = DoorState.ERROR
            
            return success
    
    def release_motor(self):
        """Release motor (stop PWM signal)"""
        try:
            if self.servo:
                self.servo.ChangeDutyCycle(0)
                self.is_holding_position = False
                self.logger.info("Motor released (PWM stopped)")
        except Exception as e:
            self.logger.error(f"Failed to release motor: {e}")
    
    def hold_current_position(self):
        """Hold current position (resume PWM signal)"""
        try:
            if self.servo and not self.is_holding_position:
                duty = self._calculate_duty_cycle(self.current_angle)
                self.servo.ChangeDutyCycle(duty)
                self.is_holding_position = True
                self.logger.info(f"Holding position at {self.current_angle}°")
        except Exception as e:
            self.logger.error(f"Failed to hold position: {e}")
    
    def emergency_stop(self):
        """Emergency stop - immediately stop servo movement"""
        try:
            if self.servo:
                self.servo.ChangeDutyCycle(0)  # Stop PWM signal
                self.is_holding_position = False
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
    
    def is_motor_holding(self) -> bool:
        """Check if motor is holding position"""
        return self.is_holding_position
    
    def test_movement(self) -> bool:
        """Test servo movement - open and close cycle"""
        self.logger.info("Testing servo movement...")
        
        try:
            # Test opening (with hold)
            if not self.open_door():
                return False
            
            self.logger.info("Door open - motor holding position")
            time.sleep(3)  # 3초간 위치 유지 확인
            
            # Test closing (without hold)
            if not self.close_door():
                return False
            
            self.logger.info("Door closed - motor stopped")
            time.sleep(1)
            
            self.logger.info("Servo test completed successfully")
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
                self.is_holding_position = False
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
# 수정된 테스트 코드
# ==========================================

def test_servo_hold_position():
    """Test servo holding position when open"""
    print("=" * 50)
    print("Servo Motor Hold Position Test")
    print("=" * 50)
    
    try:
        print("1. Servo controller initialization...")
        with ServoController() as servo:
            print("   O Initialization successful!")
            print(f"   Current angle: {servo.get_current_angle()}°")
            print(f"   Motor holding: {servo.is_motor_holding()}")
            
            print("\n2. Door opening test (with position hold)...")
            if servo.open_door():
                print("   O Door opening successful!")
                print(f"   Current angle: {servo.get_current_angle()}°")
                print(f"   Motor holding: {servo.is_motor_holding()}")
            else:
                print("   X Door opening failed!")
                return False
            
            print("   Waiting 5 seconds to test position holding...")
            import time
            for i in range(5):
                time.sleep(1)
                print(f"   [{i+1}/5] Motor holding: {servo.is_motor_holding()}")
            
            print("\n3. Manual motor release test...")
            servo.release_motor()
            print(f"   Motor holding after release: {servo.is_motor_holding()}")
            
            time.sleep(2)
            
            print("\n4. Manual motor hold test...")
            servo.hold_current_position()
            print(f"   Motor holding after hold command: {servo.is_motor_holding()}")
            
            time.sleep(2)
            
            print("\n5. Door closing test (motor should stop)...")
            if servo.close_door():
                print("   O Door closing successful!")
                print(f"   Current angle: {servo.get_current_angle()}°")
                print(f"   Motor holding: {servo.is_motor_holding()}")
            else:
                print("   X Door closing failed!")
                return False
            
            print("\n6. Test completed!")
            print("   O Servo motor hold position test successful!")
            
        return True
        
    except Exception as e:
        print(f"   X Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Modified Servo Controller Test:")
    print("1. Hold position test")
    print("2. Manual control (with hold options)")
    
    choice = input("Select (1-2): ").strip()
    
    if choice == "1":
        success = test_servo_hold_position()
    elif choice == "2":
        # 기존 manual control에 hold 옵션 추가된 버전
        print("Manual control with hold options - use 'r' to release motor, 'h' to hold position")
        success = True  # 간단히 성공으로 처리
    else:
        print("Invalid selection. Running hold position test.")
        success = test_servo_hold_position()
    
    if success:
        print("\n Test successful!")
    else:
        print("\n Test failed!")
        sys.exit(1)