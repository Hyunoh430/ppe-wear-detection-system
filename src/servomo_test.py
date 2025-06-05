import RPi.GPIO as GPIO
import time

servoPin = 2  # BCM 기준
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)
servo.start(0)

# 각도 → 듀티비 변환
def servo_control(degree):
    if degree > 180: degree = 180
    if degree < 0: degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.1)

try:
    # 현재 위치를 예: 120도로 가정
    current_degree = 120

    # 시계방향으로 닫기 (→ 각도 감소)
    for i in range(0, 31, 1):  # 1도씩 줄이기
        target_deg = current_degree - i
        print(f"닫기 동작: {target_deg}도")
        servo_control(target_deg)

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
