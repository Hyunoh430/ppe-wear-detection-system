import RPi.GPIO as GPIO
import time

# 서보모터 연결 핀 설정 (BCM 번호 기준)
servoPin = 2  # GPIO 2번

# 서보모터의 duty cycle 범위 설정
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

# 서보모터 PWM 신호 초기화 (50Hz 주파수)
servo = GPIO.PWM(servoPin, 50)
servo.start(0)

# 각도 → duty 변환 함수
def servo_control(degree, delay):
    if degree > 180:
        degree = 180
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(delay)

# 무한 루프: 닫기(90도) → 열기(150도)
try:
    while True:
        servo_control(90, 1)    # 90도로 이동 (닫기) → 1초 대기
        servo_control(150, 2)   # 150도로 이동 (열기) → 2초 대기

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
