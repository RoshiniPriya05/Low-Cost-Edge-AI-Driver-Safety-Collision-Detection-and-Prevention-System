from gpiozero import DigitalInputDevice
import time

MQ3_PIN = 17

sensor = DigitalInputDevice(MQ3_PIN)

print("MQ3 Alcohol Detection Started")

while True:

    if sensor.value == 0:
        print("⚠ Alcohol Detected!")
    else:
        print("✓ No Alcohol")

    time.sleep(1)