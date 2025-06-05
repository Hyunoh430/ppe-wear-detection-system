import RPi.GPIO as GPIO
import time

servoPin = 2  # BCM GPIO pin
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)  # SG90 uses 50Hz PWM
servo.start(0)

def servo_control(degree, delay=0.03):  # Faster speed
    if degree > 180:
        degree = 180
    if degree < 0:
        degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(delay)

try:
    # Step 1: Close more (from 20° to 120°)
    for deg in range(20, 121, 1):  # Increasing angle = clockwise
        print(f"Closing... {deg}°")
        servo_control(deg)

    time.sleep(0.5)

    # Step 2: Return to open (120° to 20°)
    for deg in range(120, 19, -1):
        print(f"Opening... {deg}°")
        servo_control(deg)

    input("Done. Press Enter to exit...")

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
