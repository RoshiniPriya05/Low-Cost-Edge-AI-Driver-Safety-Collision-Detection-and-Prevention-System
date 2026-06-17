import cv2
import time
from ultralytics import YOLO
from gpiozero import LED

# ===============================
# HARDWARE SETUP (LED ONLY)
# ===============================

LEFT_LED_PIN = 23
RIGHT_LED_PIN = 24

left_led = LED(LEFT_LED_PIN)
right_led = LED(RIGHT_LED_PIN)

# ===============================
# CAMERA SETTINGS
# ===============================

CAMERA_INDEX = 0
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Camera not detected")
    exit()

# ===============================
# AI MODEL
# ===============================

model = YOLO("yolov8n.pt")

# ===============================
# OBJECT PARAMETERS
# ===============================

IMPORTANT_OBJECTS = [
    "car","truck","bus","motorcycle","bicycle","person"
]

OBJECT_DISPLAY = {
    "person":"Person",
    "car":"Car",
    "truck":"Truck",
    "bus":"Bus",
    "motorcycle":"Bike",
    "bicycle":"Bike"
}

REAL_HEIGHT = {
    "person":1.7,
    "car":1.5,
    "truck":3.0,
    "bus":3.2,
    "motorcycle":1.4,
    "bicycle":1.4
}

FOCAL_LENGTH = 280

# ===============================
# SYSTEM PARAMETERS
# ===============================

BLIND_RATIO = 0.2
ALERT_DISTANCE = 5

prev_time = time.time()
prev_distance = None

frame_skip = 0
results = None

# ===============================
# FUNCTIONS
# ===============================

def estimate_distance(height,label):

    real = REAL_HEIGHT.get(label,1.7)

    if height == 0:
        return 999

    distance = (real * FOCAL_LENGTH) / height
    return round(distance,2)


def hazard_on(speed=0.5):

    left_led.blink(on_time=speed, off_time=speed, background=True)
    right_led.blink(on_time=speed, off_time=speed, background=True)


def hazard_off():

    left_led.off()
    right_led.off()


def get_center(x1,y1,x2,y2):

    return ((x1+x2)//2,(y1+y2)//2)


def calculate_ttc(prev_dist,curr_dist,dt):

    if prev_dist is None:
        return None

    speed = (prev_dist - curr_dist) / dt

    if speed <= 0:
        return None

    ttc = curr_dist / speed

    if ttc < 0 or ttc > 20:
        return None

    return round(ttc,2)


def blind_spot(x,width):

    left = int(width * BLIND_RATIO)
    right = int(width * (1-BLIND_RATIO))

    if x < left:
        return "LEFT"

    if x > right:
        return "RIGHT"

    return "CENTER"


def draw_radar(frame,objects):

    h,w = frame.shape[:2]

    cx = w//2
    cy = h-20

    for r in [40,80]:
        cv2.circle(frame,(cx,cy),r,(80,80,80),1)

    for obj in objects:

        x = int(cx + (obj["x"]-0.5)*100)
        y = int(cy - min(obj["distance"]*20,70))

        cv2.circle(frame,(x,y),4,(0,0,255),-1)

# ===============================
# MAIN LOOP
# ===============================

while True:

    cap.grab()
    ret,frame = cap.read()

    if not ret:
        continue

    h,w = frame.shape[:2]

    now = time.time()
    dt = now - prev_time
    prev_time = now

    frame_skip += 1

    # Run YOLO every 2 frames
    if frame_skip % 2 == 0:
        results = model(frame,conf=0.45,imgsz=320,verbose=False)

    closest = 999
    closest_obj = "None"
    radar_objects = []

    blind_status = "CLEAR"
    ttc_val = None
    count = 0

    if results:

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])
                label = model.names[cls]

                if label not in IMPORTANT_OBJECTS:
                    continue

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                height = y2-y1

                distance = estimate_distance(height,label)

                center = get_center(x1,y1,x2,y2)

                radar_objects.append({
                    "x":center[0]/w,
                    "distance":distance
                })

                if distance < closest:
                    closest = distance
                    closest_obj = OBJECT_DISPLAY.get(label,label)

                spot = blind_spot(center[0],w)

                if spot in ["LEFT","RIGHT"]:
                    blind_status = spot

                color = (0,255,0)

                if distance < 8:
                    color = (0,165,255)

                if distance < 4:
                    color = (0,0,255)

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)

                cv2.putText(frame,
                            f"{OBJECT_DISPLAY.get(label,label)} {distance:.1f}m",
                            (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            color,
                            1)

                count += 1

    ttc_val = calculate_ttc(prev_distance,closest,dt)
    prev_distance = closest

    # ===============================
    # COLLISION WARNING + LED CONTROL
    # ===============================

    if closest < 3:

        cv2.putText(frame,
                    "CRITICAL COLLISION",
                    (15,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,0,255),
                    2)

        hazard_on(0.15)

    elif closest < ALERT_DISTANCE:

        cv2.putText(frame,
                    "COLLISION WARNING",
                    (15,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,0,255),
                    2)

        hazard_on(0.5)

    else:

        hazard_off()

        # Blind spot indicators
        if blind_status == "LEFT":

            left_led.on()
            right_led.off()

        elif blind_status == "RIGHT":

            right_led.on()
            left_led.off()

        else:

            left_led.off()
            right_led.off()

    if ttc_val and ttc_val < 2.5:

        cv2.putText(frame,
                    f"IMMINENT COLLISION {ttc_val}s",
                    (15,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0,0,255),
                    1)

    if blind_status != "CLEAR":

        cv2.putText(frame,
                    f"{blind_status} BLIND SPOT",
                    (15,h-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0,255,255),
                    1)

    # ===============================
    # MINI DASHBOARD
    # ===============================

    overlay = frame.copy()

    cv2.rectangle(overlay,(8,8),(180,90),(25,25,25),-1)

    frame = cv2.addWeighted(overlay,0.6,frame,0.4,0)

    y = 25

    cv2.putText(frame,"EDGE AI SAFETY",
                (15,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,(220,220,220),1)

    y += 15

    cv2.putText(frame,f"Objects: {count}",
                (15,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,(200,200,200),1)

    y += 15

    cv2.putText(frame,f"Closest: {closest_obj}",
                (15,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,(200,200,200),1)

    y += 15

    dist_text = f"{closest:.1f}m" if closest < 999 else "N/A"

    cv2.putText(frame,f"Dist: {dist_text}",
                (15,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,(200,200,200),1)

    y += 15

    ttc_text = f"{ttc_val}s" if ttc_val else "N/A"

    cv2.putText(frame,f"TTC: {ttc_text}",
                (15,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,(200,200,200),1)

    # ===============================
    # FPS
    # ===============================

    fps = 1/max(dt,0.001)

    cv2.putText(frame,
                f"FPS:{int(fps)}",
                (w-80,20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255,255,255),
                1)

    draw_radar(frame,radar_objects)

    cv2.imshow("Edge AI Driver Safety System",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()