import RPi.GPIO as GPIO
import time

servoPin = 2  # BCM pin
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)
servo.start(0)

def servo_control(degree, delay=0.1):
    if degree > 180:
        degree = 180
    if degree < 0:
        degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(delay)

try:
    # Assume current position is 150° (open)
    current_degree = 150

    # Move clockwise → decrease angle (e.g., to 90°)
    for deg in range(current_degree, 89, -1):  # 150 → 90
        print(f"Closing... {deg}°")
        servo_control(deg)

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
