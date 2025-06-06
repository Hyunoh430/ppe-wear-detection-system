import RPi.GPIO as GPIO
import time

servoPin = 2
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.cleanup()
time.sleep(0.2)

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)  # SG90 uses 50Hz
servo.start(0)

def servo_control(degree, delay=0.03):
    if degree > 180: degree = 180
    if degree < 0: degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(delay)

try:
    # 닫힌 상태에서 열린 상태로 이동 (100 → 10)
    print("열기 중...")
    for deg in range(100, 9, -1):
        servo_control(deg)
    print("열림 완료")
    
    time.sleep(1.5)

    # 열린 상태에서 닫힌 상태로 이동 (10 → 100)
    print("닫기 중...")
    for deg in range(10, 101):
        servo_control(deg)
    print("닫힘 완료")

    input("Done. Press Enter to exit...")

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
