import RPi.GPIO as GPIO
import time

servoPin = 2  # GPIO 번호 (BCM 기준)
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)
servo.start(0)

def servo_control(degree):
    if degree > 180:
        degree = 180
    if degree < 0:
        degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.1)  # 천천히 움직이게

try:
    for deg in range(150, 89, -1):  # 150 → 90도, 1도씩 감소
        print(f"각도: {deg}도")
        servo_control(deg)

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
