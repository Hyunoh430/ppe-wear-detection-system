import RPi.GPIO as GPIO
import time

servoPin = 2  # BCM 기준 GPIO 핀 번호
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)  # 50Hz 주기
servo.start(0)

def servo_control(degree, delay=1.5):
    if degree > 180:
        degree = 180
    if degree < 0:
        degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(delay)

try:
    # Move servo to 30 degrees (opposite of 150)
    print("Moving servo to 30 degrees...")
    servo_control(20)

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
