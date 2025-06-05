import RPi.GPIO as GPIO
import time
import numpy as np

servoPin = 2
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)
servo.start(0)

# 서보 각도 제어 함수 (각도에 따라 속도 다르게)
def servo_control_with_variable_speed(degree):
    if degree > 180: degree = 180
    if degree < 0: degree = 0

    # 중간 구간(70~100도)은 더 천천히
    if 70 <= degree <= 100:
        delay = 0.15
    else:
        delay = 0.07

    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(delay)

try:
    # 닫힘 동작: 30° → 130°
    for deg in np.arange(30, 131, 0.5):
        print(f"Closing... {deg:.1f}°")
        servo_control_with_variable_speed(deg)

    time.sleep(0.5)

    # 열림 동작: 130° → 30° (중간 구간만 느리게)
    for deg in np.arange(130, 29, -0.5):
        print(f"Opening... {deg:.1f}°")
        servo_control_with_variable_speed(deg)

    input("Done. Press Enter to exit...")

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
