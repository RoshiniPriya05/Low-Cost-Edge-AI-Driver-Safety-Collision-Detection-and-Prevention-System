import cv2
import time
import numpy as np
from gpiozero import DigitalInputDevice

# ==============================
# MQ3 ALCOHOL SENSOR SETUP
# ==============================

ALCOHOL_PIN = 17
sensor = DigitalInputDevice(ALCOHOL_PIN)

# ==============================
# LOAD HAAR CASCADES
# ==============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# ==============================
# DISPLAY MESSAGE FUNCTION
# ==============================

def show_message(text, color=(0,0,0), duration=2000):

    screen = np.ones((350,600,3), dtype=np.uint8) * 255

    cv2.putText(screen,
                text,
                (70,180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2)

    cv2.imshow("Driver Safety System", screen)
    cv2.waitKey(duration)


# ==============================
# BLOW COUNTDOWN
# ==============================

def countdown():

    for i in range(3,0,-1):

        screen = np.ones((350,600,3), dtype=np.uint8) * 255

        cv2.putText(screen,
                    "Blow Near Sensor",
                    (150,130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,0),
                    2)

        cv2.putText(screen,
                    str(i),
                    (290,230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0,0,255),
                    3)

        cv2.imshow("Driver Safety System", screen)
        cv2.waitKey(1000)


# ==============================
# ALCOHOL CHECK USING MQ3
# ==============================

def alcohol_check():

    print("Starting Alcohol Test")

    show_message("Alcohol Test Starting")

    countdown()

    detected = False
    start = time.time()

    show_message("Checking...", (0,0,0), 1000)

    while time.time() - start < 4:

        if sensor.value == 0:   # MQ3 LOW means alcohol detected
            detected = True

    time.sleep(0.1)

    if detected:

        show_message("ALCOHOL DETECTED", (0,0,255), 3000)
        show_message("DRIVING NOT ALLOWED", (0,0,255), 3000)

        print("Alcohol detected")

        return False

    else:

        show_message("ALCOHOL TEST PASSED", (0,180,0), 3000)

        print("No alcohol detected")

        return True


# ==============================
# DRIVER ALERTNESS TEST
# ==============================

def alertness_test():

    print("Starting Driver Alertness Test")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not detected")
        return False

    start_time = time.time()

    face_frames = 0
    eye_frames = 0
    total_frames = 0

    while time.time() - start_time < 10:

        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        total_frames += 1

        for (x,y,w,h) in faces:

            face_frames += 1

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) >= 1:
                eye_frames += 1

            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

        cv2.putText(frame,
                    "Driver Alertness Scan",
                    (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2)

        cv2.imshow("Driver Alertness", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if total_frames == 0:
        return False

    face_score = (face_frames / total_frames) * 40
    eye_score = (eye_frames / total_frames) * 60

    score = int(face_score + eye_score)

    print("Driver Alertness Score:", score)

    if score >= 70:

        print("Driver is ALERT")

        return True

    else:

        print("Driver NOT ALERT")

        return False


# ==============================
# MAIN SYSTEM
# ==============================

def driver_precheck():

    # STEP 1 → Alcohol Test

    if not alcohol_check():

        print("System stopped due to alcohol detection")

        return

    # STEP 2 → Driver Alertness

    result = alertness_test()

    if result:

        show_message("Driver Ready To Drive", (0,200,0), 3000)

        print("Driver cleared readiness test")

    else:

        show_message("Driver Not Fit To Drive", (0,0,255), 3000)

        print("Driver failed readiness test")


# ==============================
# RUN SYSTEM
# ==============================

if __name__ == "__main__":

    driver_precheck()