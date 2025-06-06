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
    # Open: 100 → 90 (faster)
    for deg in range(100, 89, -1):
        servo_control(deg, delay=0.005)

    time.sleep(1.2)

    # Close: 90 → 60 (slow, controlled)
    for deg in range(90, 59, -1):
        servo_control(deg, delay=0.03)

    # Release for gravity-assisted final close
    servo.ChangeDutyCycle(0)
    time.sleep(1.0)

    # Return to 90
    for deg in range(60, 91):
        servo_control(deg, delay=0.02)

    input("Done. Press Enter to exit...")

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
