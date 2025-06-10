import RPi.GPIO as GPIO
import time

servoPin = 2
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.cleanup()
time.sleep(0.2)

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)
servo.start(0)

def servo_control(degree, delay=0.01):
    if degree > 180: degree = 180
    if degree < 0: degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(delay)

try:
    # Open: 100 → 70 (fast)
    for deg in range(100, 69, -1):  # 100에서 70까지 빠르게 이동
        servo_control(deg, delay=0.005)

    # Hold at 70 degrees for 2 seconds
    hold_duty = SERVO_MIN_DUTY + (70 * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(hold_duty)
    time.sleep(2.0)

    # Close: 70 → 100 (slow)
    for deg in range(70, 101):  # 70에서 100까지 천천히 이동
        servo_control(deg, delay=0.03)

    input("Done. Press Enter to exit...")

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()