"""
Servo motor controller for waste disposal door
"""

import RPi.GPIO as GPIO
import time
import threading
import logging
from typing import Optional
from enum import Enum

from .config import *

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
            
            # Set initial position (closed)
            self._set_angle_immediate(SERVO_CLOSED_ANGLE)
            
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
    
    def _set_angle_immediate(self, angle: float):
        """Set servo to specific angle immediately"""
        if not self.is_initialized or self.servo is None:
            self.logger.error("Servo not initialized")
            return False
        
        try:
            duty = self._calculate_duty_cycle(angle)
            self.servo.ChangeDutyCycle(duty)
            self.current_angle = angle
            time.sleep(SERVO_MOVE_DELAY)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set servo angle: {e}")
            self.state = DoorState.ERROR
            return False
    
    def _move_to_angle_smooth(self, target_angle: float, speed: int = 1):
        """Move servo smoothly to target angle"""
        if not self.is_initialized:
            return False
        
        self.state = DoorState.MOVING
        current = int(self.current_angle)
        target = int(target_angle)
        
        if current < target:
            step = speed
            angle_range = range(current, target + 1, step)
        else:
            step = -speed
            angle_range = range(current, target - 1, step)
        
        try:
            for angle in angle_range:
                if not self._set_angle_immediate(angle):
                    return False
                
            # Ensure we reach exact target
            self._set_angle_immediate(target_angle)
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
                success = self._set_angle_immediate(SERVO_OPEN_ANGLE)
            
            if success:
                self.state = DoorState.OPEN
                self.logger.info("Door opened successfully")
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
                success = self._set_angle_immediate(SERVO_CLOSED_ANGLE)
            
            if success:
                self.state = DoorState.CLOSED
                self.logger.info("Door closed successfully")
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
        """Test servo movement - open and close cycle"""
        self.logger.info("Testing servo movement...")
        
        try:
            # Test opening
            if not self.open_door():
                return False
            
            time.sleep(1)
            
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
    """기본 서보모터 테스트"""
    print("=" * 50)
    print("서보모터 기본 테스트")
    print("=" * 50)
    
    try:
        print("1. 서보 컨트롤러 초기화...")
        with ServoController() as servo:
            print("   ✓ 초기화 성공!")
            print(f"   현재 각도: {servo.get_current_angle()}°")
            print(f"   도어 상태: {servo.get_door_state().value}")
            
            print("\n2. 도어 열기 테스트...")
            if servo.open_door():
                print("   ✓ 도어 열기 성공!")
                print(f"   현재 각도: {servo.get_current_angle()}°")
                print(f"   도어 상태: {servo.get_door_state().value}")
            else:
                print("   ✗ 도어 열기 실패!")
                return False
            
            import time
            print("   3초 대기...")
            time.sleep(3)
            
            print("\n3. 도어 닫기 테스트...")
            if servo.close_door():
                print("   ✓ 도어 닫기 성공!")
                print(f"   현재 각도: {servo.get_current_angle()}°")
                print(f"   도어 상태: {servo.get_door_state().value}")
            else:
                print("   ✗ 도어 닫기 실패!")
                return False
            
            print("\n4. 테스트 완료!")
            print("   ✓ 서보모터 정상 작동!")
            
        return True
        
    except Exception as e:
        print(f"   ✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_servo_manual_control():
    """수동 서보 제어 테스트"""
    print("=" * 50)
    print("서보모터 수동 제어 테스트")
    print("=" * 50)
    print("명령어:")
    print("  o - 도어 열기")
    print("  c - 도어 닫기") 
    print("  s - 상태 확인")
    print("  t - 전체 테스트")
    print("  q - 종료")
    print("=" * 50)
    
    try:
        with ServoController() as servo:
            print(f"초기 상태: {servo.get_door_state().value}")
            
            while True:
                command = input("\n명령어 입력: ").strip().lower()
                
                if command == 'q':
                    print("테스트 종료!")
                    break
                elif command == 'o':
                    print("도어 열기...")
                    if servo.open_door():
                        print("✓ 도어 열림!")
                    else:
                        print("✗ 도어 열기 실패!")
                elif command == 'c':
                    print("도어 닫기...")
                    if servo.close_door():
                        print("✓ 도어 닫힘!")
                    else:
                        print("✗ 도어 닫기 실패!")
                elif command == 's':
                    print(f"현재 상태:")
                    print(f"  - 도어 상태: {servo.get_door_state().value}")
                    print(f"  - 현재 각도: {servo.get_current_angle()}°")
                    print(f"  - 도어 열림: {servo.is_door_open()}")
                    print(f"  - 도어 닫힘: {servo.is_door_closed()}")
                elif command == 't':
                    print("전체 움직임 테스트...")
                    if servo.test_movement():
                        print("✓ 전체 테스트 성공!")
                    else:
                        print("✗ 전체 테스트 실패!")
                else:
                    print("알 수 없는 명령어입니다.")
                    
        return True
        
    except KeyboardInterrupt:
        print("\n사용자가 테스트를 중단했습니다.")
        return True
    except Exception as e:
        print(f"✗ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_servo_angles():
    """다양한 각도 테스트"""
    print("=" * 50)
    print("서보모터 각도 테스트")
    print("=" * 50)
    
    try:
        with ServoController() as servo:
            test_angles = [0, 30, 60, 90, 120, 150, 180]
            
            print("다양한 각도로 서보 이동 테스트...")
            
            for angle in test_angles:
                print(f"  {angle}° 이동 중...")
                servo._set_angle_immediate(angle)
                
                import time
                time.sleep(1)
                
                print(f"    현재 각도: {servo.get_current_angle()}°")
            
            print("\n원래 위치(20°)로 복귀...")
            servo._set_angle_immediate(20)
            
            print("✓ 각도 테스트 완료!")
            
        return True
        
    except Exception as e:
        print(f"✗ 오류: {e}")
        return False

def check_gpio_setup():
    """GPIO 설정 확인"""
    print("=" * 50)
    print("GPIO 설정 확인")
    print("=" * 50)
    
    try:
        import RPi.GPIO as GPIO
        
        print("1. RPi.GPIO 라이브러리 확인... ✓")
        
        print("2. GPIO 권한 확인...")
        GPIO.setmode(GPIO.BCM)
        print("   ✓ GPIO 설정 가능!")
        
        print("3. PWM 테스트...")
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        pwm = GPIO.PWM(SERVO_PIN, 50)
        pwm.start(0)
        print("   ✓ PWM 초기화 성공!")
        
        pwm.stop()
        GPIO.cleanup()
        
        print("✓ GPIO 설정 모두 정상!")
        return True
        
    except Exception as e:
        print(f"✗ GPIO 오류: {e}")
        print("\n해결방법:")
        print("1. sudo로 실행: sudo python servo_controller.py")
        print("2. 사용자를 gpio 그룹에 추가: sudo usermod -a -G gpio $USER")
        return False

if __name__ == "__main__":
    import sys
    
    print("서보모터 컨트롤러 테스트 옵션:")
    print("1. 기본 테스트 (열기/닫기)")
    print("2. 수동 제어 테스트")
    print("3. 각도 테스트")
    print("4. GPIO 설정 확인")
    
    choice = input("선택 (1-4): ").strip()
    
    if choice == "1":
        success = test_servo_basic()
    elif choice == "2":
        success = test_servo_manual_control()
    elif choice == "3":
        success = test_servo_angles()
    elif choice == "4":
        success = check_gpio_setup()
    else:
        print("잘못된 선택입니다. 기본 테스트를 실행합니다.")
        success = test_servo_basic()
    
    if success:
        print("\n🎉 테스트 성공!")
    else:
        print("\n❌ 테스트 실패!")
        sys.exit(1)