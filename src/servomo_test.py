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
    # 닫기: 20 → 130 (빠→느→빠)
    for deg in range(20, 61):
        servo_control(deg, delay=0.015)

    for deg in range(61, 101):
        servo_control(deg, delay=0.04)

    for deg in range(101, 131):
        servo_control(deg, delay=0.015)

    time.sleep(0.5)

    # 열기: 130 → 30 (균일하게)
    for deg in range(130, 29, -1):
        servo_control(deg, delay=0.03)

    input("Done. Press Enter to exit...")
    
except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
