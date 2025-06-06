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

def servo_control(degree, delay):
    if degree > 180: degree = 180
    if degree < 0: degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(delay)

try:
    # open partially (100 → 40)
    for deg in range(100, 39, -1):
        servo_control(deg, delay=0.01)

    # small pause to let lid stay open briefly
    time.sleep(1)

    # release PWM signal (let gravity close lid)
    servo.ChangeDutyCycle(0)
    print("released for gravity-based closing")

    input("Press Enter to exit...")

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
