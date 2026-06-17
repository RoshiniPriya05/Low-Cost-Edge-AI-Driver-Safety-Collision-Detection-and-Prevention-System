import cv2
import time
import collections
import threading
import numpy as np
import mediapipe as mp

# ================= SENSOR SETUP =================
try:
    from gpiozero import Button
    DOUT_GPIO = 17
    mq3 = Button(DOUT_GPIO, pull_up=True)
    SENSOR_AVAILABLE = True
    print("[MQ3] Digital alcohol sensor active on GPIO17")
except:
    mq3 = None
    SENSOR_AVAILABLE = False
    print("[MQ3] Sensor not detected - camera mode only")

# ================= PARAMETERS =================
WARN_THRESHOLD = 0.40
ALERT_THRESHOLD = 0.60
WINDOW = 60

EAR_VAR_THRESH = 0.0012
HEAD_SWAY_THRESH = 8.0
MAR_SPIKE_THRESH = 0.55
BLINK_RATE_LOW = 5
BLINK_RATE_HIGH = 35

# ================= GLOBAL STATE =================
ear_window = collections.deque(maxlen=WINDOW)
mar_window = collections.deque(maxlen=WINDOW)
nose_window = collections.deque(maxlen=WINDOW)
blink_times = collections.deque(maxlen=30)

last_eye_open = True

drunk_score = 0.0
drunk_warning = False
drunk_detected = False

lock = threading.Lock()

# ================= MEDIAPIPE SETUP =================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# ================= CAMERA =================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ================= UTIL FUNCTIONS =================

def eye_aspect_ratio(landmarks, left=True):
    if left:
        pts = [33,160,158,133,153,144]
    else:
        pts = [362,385,387,263,373,380]

    p = [np.array(landmarks[i]) for i in pts]
    A = np.linalg.norm(p[1]-p[5])
    B = np.linalg.norm(p[2]-p[4])
    C = np.linalg.norm(p[0]-p[3])

    return (A+B)/(2*C)

def mouth_aspect_ratio(landmarks):

    pts = [13,14,78,308,82,312,87,317]
    p = [np.array(landmarks[i]) for i in pts]

    A = np.linalg.norm(p[0]-p[1])
    B = np.linalg.norm(p[2]-p[3])
    C = np.linalg.norm(p[4]-p[5])
    D = np.linalg.norm(p[6]-p[7])

    return (A+B+C+D)/4

# ================= CAMERA SCORE =================

def compute_camera_score():

    score = 0
    signals = 0

    if len(ear_window) > 2:
        import statistics
        ear_var = statistics.variance(ear_window)
        if ear_var > EAR_VAR_THRESH:
            score += min(ear_var/(EAR_VAR_THRESH*4),1)*0.35
            signals += 1

    if len(nose_window) > 2:
        import statistics
        sway = statistics.stdev(nose_window)
        if sway > HEAD_SWAY_THRESH:
            score += min(sway/(HEAD_SWAY_THRESH*3),1)*0.35
            signals += 1

    if len(mar_window) > 2:
        spikes = sum(1 for m in mar_window if m>MAR_SPIKE_THRESH)
        if spikes>0:
            score += min(spikes/10,1)*0.2
            signals += 1

    now = time.time()
    bpm = len([t for t in blink_times if now-t<=60])

    if bpm < BLINK_RATE_LOW or bpm > BLINK_RATE_HIGH:
        score += 0.1
        signals += 1

    if signals < 2:
        score *= 0.4

    return min(score,1)

# ================= SENSOR READ =================

def read_mq3():

    if not SENSOR_AVAILABLE:
        return False

    return mq3.is_pressed

# ================= STATE MACHINE =================

WAIT_DRIVER = 0
CHECK_DRUNK = 1
FITNESS_TEST = 2

state_machine = WAIT_DRIVER

# ================= MAIN LOOP =================

frame_skip = 0

while True:

    ret, frame = cap.read()
    if not ret:
        continue

    frame_skip += 1
    if frame_skip % 2 != 0:
        continue

    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    EAR = 0.3
    MAR = 0
    nose_x = w//2
    eye_open = True
    driver_present = False

    if results.multi_face_landmarks:

        driver_present = True

        landmarks = results.multi_face_landmarks[0].landmark
        pts = [(int(l.x*w), int(l.y*h)) for l in landmarks]

        EAR = (eye_aspect_ratio(pts,True)+eye_aspect_ratio(pts,False))/2
        MAR = mouth_aspect_ratio(pts)
        nose_x = pts[1][0]

        eye_open = EAR > 0.25

        mp_draw.draw_landmarks(frame, results.multi_face_landmarks[0], mp_face.FACEMESH_TESSELATION)

    with lock:

        ear_window.append(EAR)
        mar_window.append(MAR)
        nose_window.append(nose_x)

        if last_eye_open and not eye_open:
            blink_times.append(time.time())

    last_eye_open = eye_open

    cam_score = compute_camera_score()
    sensor_hit = read_mq3()

    if sensor_hit:
        drunk_score = max(cam_score,0.85)
        drunk_detected = True
        drunk_warning = True
    else:
        drunk_score = cam_score
        drunk_warning = cam_score >= WARN_THRESHOLD
        drunk_detected = cam_score >= ALERT_THRESHOLD

    # ================= STATE MACHINE =================

    if state_machine == WAIT_DRIVER:

        if driver_present:
            state_machine = CHECK_DRUNK

    elif state_machine == CHECK_DRUNK:

        if drunk_detected:
            cv2.putText(frame,"DRUNK DETECTED",(20,70),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            state_machine = FITNESS_TEST

    elif state_machine == FITNESS_TEST:

        cv2.putText(frame,"FITNESS TEST RUNNING",(20,70),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    cv2.putText(frame,f"Drunk Score: {drunk_score:.2f}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Driver Monitor",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()