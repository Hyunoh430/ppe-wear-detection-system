import RPi.GPIO as GPIO
import time

servoPin = 2
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# 안전하게 초기화
GPIO.cleanup()
time.sleep(0.2)

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

# PWM 객체 생성 (주파수 50Hz)
servo = GPIO.PWM(servoPin, 50)
servo.start(0)

def servo_control(degree):
    if degree > 180: degree = 180
    if degree < 0: degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    print(f"Angle: {degree}°, Duty: {round(duty, 2)}")
    time.sleep(0.05)

try:
    # 0 → 180도 (1도씩 증가)
    for deg in range(0, 181):
        servo_control(deg)

    time.sleep(1)

    # 180 → 0도 (1도씩 감소)
    for deg in range(180, -1, -1):
        servo_control(deg)

    input("Done. Press Enter to exit...")

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
