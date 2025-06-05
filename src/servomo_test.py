import RPi.GPIO as GPIO
import time

servoPin = 2  # Using BCM pin 2
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)  # 50Hz for servo
servo.start(0)

def servo_control(degree):
    if degree > 180:
        degree = 180
    if degree < 0:
        degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.1)  # Delay between each step

try:
    # Assume current position (set manually before running)
    current_degree = 120

    # Move slowly in closing direction (clockwise → increasing angle)
    for i in range(0, 21, 1):  # Move 20 degrees forward
        target_deg = current_degree + i
        print(f"Closing... Moving to {target_deg} degrees")
        servo_control(target_deg)

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
