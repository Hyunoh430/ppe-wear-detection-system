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

def servo_control(degree, delay):
    if degree > 180: degree = 180
    if degree < 0: degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(delay)

try:
    # ⏬ 닫힘 (30 → 130): 빠르게
    for deg in np.arange(30, 131, 0.5):
        print(f"Closing... {deg:.1f}°")
        servo_control(deg, delay=0.03)  # 빠른 속도

    time.sleep(0.5)

    # ⏫ 열림 (130 → 30): 중간 구간만 느리게
    for deg in np.arange(130, 29, -0.5):
        if 70 <= deg <= 100:
            delay = 0.15  # 중간 부하 구간 느리게
        else:
            delay = 0.07  # 나머지는 기본 속도
        print(f"Opening... {deg:.1f}°")
        servo_control(deg, delay)

    input("Done. Press Enter to exit...")

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
