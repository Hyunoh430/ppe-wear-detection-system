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
    # Close: 20° → 120° (빠르게 가능)
    for deg in np.arange(20, 130, 0.5):
        servo_control(deg, delay=0.03)

    time.sleep(0.5)

    # Open: 120° → 30° (느리게, 더 여유 있게)
    for deg in np.arange(130, 29, -0.5):
        servo_control(deg, delay=0.08)  # 느리게, 부드럽게

    input("Done. Press Enter to exit...")

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
