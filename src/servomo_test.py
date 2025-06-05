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

def servo_control(degree, delay=0.05):
    if degree > 180: degree = 180
    if degree < 0: degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(delay)

try:
    # 닫기 (20° → 130°): 초반 빠르게 → 중간 느리게 → 후반 빠르게

    # Step 1: 20 → 60 (빠르게)
    for deg in range(20, 61):
        servo_control(deg, delay=0.015)

    # Step 2: 61 → 100 (중간을 약하게, 느리게)
    for deg in range(61, 101):
        servo_control(deg, delay=0.04)  # 부드럽게 밀기

    # Step 3: 101 → 130 (빠르게 마무리)
    for deg in range(101, 131):
        servo_control(deg, delay=0.015)

    time.sleep(0.5)

    # 열기 (130 → 30): 초반 부드럽게 시작 → 이후 빠르게

    for deg in range(130, 121, -1):  # 초반 부드럽게
        servo_control(deg, delay=0.015)

    for deg in range(121, 59, -1):   # 중간 빠르게
        servo_control(deg, delay=0.015)

    for deg in range(59, 29, -1):    # 후반 빠르게
        servo_control(deg, delay=0.015)


    input("Done. Press Enter to exit...")

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
