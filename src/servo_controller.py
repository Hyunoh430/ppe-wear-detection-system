"""
Servo motor controller for waste disposal door
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
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            
            self.servo = GPIO.PWM(self.pin, SERVO_FREQUENCY)
            self.servo.start(0)
            
            # Set initial position (closed) and stop PWM
            self._set_angle_and_stop(SERVO_CLOSED_ANGLE)
            
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
    
    def _set_angle_and_stop(self, angle: float, hold_time: float = 0.5):
        """Set servo to specific angle and stop PWM signal"""
        if not self.is_initialized or self.servo is None:
            self.logger.error("Servo not initialized")
            return False
        
        try:
            duty = self._calculate_duty_cycle(angle)
            
            # Send PWM signal to move servo
            self.servo.ChangeDutyCycle(duty)
            
            # Hold position for specified time to ensure movement completion
            time.sleep(hold_time)
            
            # Stop PWM signal to prevent continuous rotation
            self.servo.ChangeDutyCycle(0)
            
            self.current_angle = angle
            self.logger.debug(f"Servo moved to {angle}° and PWM stopped")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set servo angle: {e}")
            self.state = DoorState.ERROR
            return False
    
    def _move_to_angle_smooth(self, target_angle: float, speed: int = 2):
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
            
            # Final position with longer hold and PWM stop
            self._set_angle_and_stop(target_angle, hold_time=0.8)
            self.target_angle = target_angle
            
            return True
            
        except Exception as e:
            self.logger.error(f"Smooth movement failed: {e}")
            self.state = DoorState.ERROR
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
            
            if smooth:
                success = self._move_to_angle_smooth(SERVO_OPEN_ANGLE)
            else:
                success = self._set_angle_and_stop(SERVO_OPEN_ANGLE)
            
            if success:
                self.state = DoorState.OPEN
                self.logger.info(f"Door opened successfully (angle: {self.current_angle}°)")
            else:
                self.logger.error("Failed to open door")
                self.state = DoorState.ERROR
            
            return success
    
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
            
            if smooth:
                success = self._move_to_angle_smooth(SERVO_CLOSED_ANGLE)
            else:
                success = self._set_angle_and_stop(SERVO_CLOSED_ANGLE)
            
            if success:
                self.state = DoorState.CLOSED
                self.logger.info(f"Door closed successfully (angle: {self.current_angle}°)")
            else:
                self.logger.error("Failed to close door")
                self.state = DoorState.ERROR
            
            return success
    
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
        """Test servo movement - open and close cycle (테스트용으로만 사용)"""
        self.logger.info("Testing servo movement...")
        
        try:
            # Test opening
            if not self.open_door():
                return False
            
            time.sleep(2)
            
            # Test closing
            if not self.close_door():
                return False
            
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
# 개별 테스트 코드
# ==========================================

def test_servo_basic():
    """Basic servo motor test"""
    print("=" * 50)
    print("Servo Motor Basic Test")
    print("=" * 50)
    
    try:
        print("1. Servo controller initialization...")
        with ServoController() as servo:
            print("   O Initialization successful!")
            print(f"   Current angle: {servo.get_current_angle()}°")
            print(f"   Door state: {servo.get_door_state().value}")
            
            print("\n2. Door opening test...")
            if servo.open_door():
                print("   O Door opening successful!")
                print(f"   Current angle: {servo.get_current_angle()}°")
                print(f"   Door state: {servo.get_door_state().value}")
            else:
                print("   X Door opening failed!")
                return False
            
            import time
            print("   Waiting 3 seconds...")
            time.sleep(3)
            
            print("\n3. Door closing test...")
            if servo.close_door():
                print("   O Door closing successful!")
                print(f"   Current angle: {servo.get_current_angle()}°")
                print(f"   Door state: {servo.get_door_state().value}")
            else:
                print("   X Door closing failed!")
                return False
            
            print("\n4. Test completed!")
            print("   O Servo motor working normally!")
            
        return True
        
    except Exception as e:
        print(f"   X Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_servo_manual_control():
    """Manual servo control test"""
    print("=" * 50)
    print("Servo Motor Manual Control Test")
    print("=" * 50)
    print("Commands:")
    print("  o - Open door")
    print("  c - Close door") 
    print("  s - Check status")
    print("  t - Full test")
    print("  q - Quit")
    print("=" * 50)
    
    try:
        with ServoController() as servo:
            print(f"Initial state: {servo.get_door_state().value}")
            
            while True:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'q':
                    print("Test terminated!")
                    break
                elif command == 'o':
                    print("Opening door...")
                    if servo.open_door():
                        print("O Door opened!")
                    else:
                        print("X Door opening failed!")
                elif command == 'c':
                    print("Closing door...")
                    if servo.close_door():
                        print("O Door closed!")
                    else:
                        print("X Door closing failed!")
                elif command == 's':
                    print(f"Current status:")
                    print(f"  - Door state: {servo.get_door_state().value}")
                    print(f"  - Current angle: {servo.get_current_angle()}°")
                    print(f"  - Door open: {servo.is_door_open()}")
                    print(f"  - Door closed: {servo.is_door_closed()}")
                elif command == 't':
                    print("Running full movement test...")
                    if servo.test_movement():
                        print("O Full test successful!")
                    else:
                        print("X Full test failed!")
                else:
                    print("Unknown command.")
                    
        return True
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        return True
    except Exception as e:
        print(f"X Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_servo_angles():
    """Various angle test"""
    print("=" * 50)
    print("Servo Motor Angle Test")
    print("=" * 50)
    
    try:
        with ServoController() as servo:
            test_angles = [0, 30, 60, 90, 120, 150, 180]
            
            print("Testing servo movement with various angles...")
            
            for angle in test_angles:
                print(f"  Moving to {angle}°...")
                servo._set_angle_and_stop(angle)
                
                import time
                time.sleep(2)  # 더 긴 대기 시간
                
                print(f"    Current angle: {servo.get_current_angle()}°")
            
            print("\nReturning to original position (120°)...")
            servo._set_angle_and_stop(120)  # 수정: 닫힌 위치로 복귀
            
            print("O Angle test completed!")
            
        return True
        
    except Exception as e:
        print(f"X Error: {e}")
        return False

def check_gpio_setup():
    """Check GPIO setup"""
    print("=" * 50)
    print("GPIO Setup Check")
    print("=" * 50)
    
    try:
        import RPi.GPIO as GPIO
        
        print("1. RPi.GPIO library check... O")
        
        print("2. GPIO permission check...")
        GPIO.setmode(GPIO.BCM)
        print("   O GPIO setup available!")
        
        print("3. PWM test...")
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        pwm = GPIO.PWM(SERVO_PIN, 50)
        pwm.start(0)
        print("   O PWM initialization successful!")
        
        pwm.stop()
        GPIO.cleanup()
        
        print("O All GPIO setup is normal!")
        return True
        
    except Exception as e:
        print(f"X GPIO error: {e}")
        print("\nSolutions:")
        print("1. Run with sudo: sudo python servo_controller.py")
        print("2. Add user to gpio group: sudo usermod -a -G gpio $USER")
        return False

if __name__ == "__main__":
    import sys
    
    print("Servo Motor Controller Test Options:")
    print("1. Basic test (open/close)")
    print("2. Manual control test")
    print("3. Angle test")
    print("4. GPIO setup check")
    
    choice = input("Select (1-4): ").strip()
    
    if choice == "1":
        success = test_servo_basic()
    elif choice == "2":
        success = test_servo_manual_control()
    elif choice == "3":
        success = test_servo_angles()
    elif choice == "4":
        success = check_gpio_setup()
    else:
        print("Invalid selection. Running basic test.")
        success = test_servo_basic()
    
    if success:
        print("\n Test successful!")
    else:
        print("\n Test failed!")
        sys.exit(1)