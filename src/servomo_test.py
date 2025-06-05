import RPi.GPIO as GPIO
import time

servoPin = 2  # BCM pin number
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)
servo.start(0)

def servo_control(degree, delay=0.05):
    if degree > 180:
        degree = 180
    if degree < 0:
        degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(delay)

try:
    # Step 1: Closing (20 → 90)
    current_degree = 20
    for deg in range(current_degree, 91, 1):  # 시계방향: 증가
        print(f"Closing... {deg}°")
        servo_control(deg)

    time.sleep(0.5)  # Short pause

    # Step 2: Re-open (90 → 20)
    print("Returning to open position (20°)")
    for deg in range(90, 19, -1):  # 반시계방향: 감소
        print(f"Opening... {deg}°")
        servo_control(deg)

    input("Done. Press Enter to exit...")

except KeyboardInterrupt:
    pass

finally:
    servo.stop()
    GPIO.cleanup()
