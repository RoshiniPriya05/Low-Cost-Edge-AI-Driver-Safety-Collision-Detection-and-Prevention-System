"""
app.py — Edge AI Driver Safety System v2
=========================================
Flow:
  Power on
    → relay SOFTWARE-LOCKED (awaiting fit test)
    → FIT_TEST mode  (6-phase pre-drive test)
         PASS → MONITORING mode (fatigue + drunk detection active)
                  fatigue 3/5 → buzzer warning
                  fatigue 5/5 → buzzer + Bolna call + WhatsApp to supervisor
                  drunk  40%  → buzzer warning
                  drunk  60%  → buzzer + Bolna call + WhatsApp to supervisor
                  road cam activates on first fatigue event
         FAIL → LOCKED (buzzer + Bolna call + WhatsApp, retry button on dashboard)

Cameras:
  Driver : Pi CSI (picamera2)
  Road   : USB webcam (auto-detected)

Alcohol:
  MQ-3 via MCP3008 SPI (drunk_detector.py handles all ADC reads)
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import time
import threading
import os
from math import hypot
from queue import Queue

from flask import Flask, Response, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from picamera2 import Picamera2

import fit_test       as ft
import drunk_detector as dd
import notifier       as notif
import ignition_relay as relay

try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    yolo_available = False
    print("[WARN] ultralytics not installed — road detection disabled")

try:
    import lgpio
    from gpiozero import Button, Buzzer, LED, Device
    from gpiozero.pins.lgpio import LGPIOFactory
    gpio_available = True
except ImportError as _e:
    gpio_available = False
    print(f"[WARN] GPIO not available ({_e})")

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode,
    )
    mp_available = True
except ImportError:
    mp_available = False
    print("[WARN] mediapipe not installed — driver detection disabled")

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
STREAM_WIDTH   = 640
STREAM_HEIGHT  = 480
STREAM_FPS     = 15
PORT           = 5050

YOLO_MODEL_PATH = "yolov8n.pt"
FACE_TASK_PATH  = "face_landmarker.task"

DRIVER_ID = os.getenv("DRIVER_ID", "Driver-001")

# Fatigue thresholds
EAR_THRESHOLD        = 0.20
MAR_THRESHOLD        = 0.35
EYE_FRAMES           = 20
YAWN_FRAMES          = 10
HEAD_NOD_FRAMES      = 10
COMBINED_EVENT_LIMIT = 5
COMBINED_WARN_LEVEL  = 3    # buzzer warning at this many events
HEAD_DROP_THRESHOLD  = 0.65
ROAD_CAM_TRIGGER     = 1

# Collision / distance
IMPORTANT_OBJECTS = ["car","truck","bus","motorcycle","bicycle","person"]
OBJECT_DISPLAY    = {"person":"Person","car":"Car","truck":"Truck",
                     "bus":"Bus","motorcycle":"Bike","bicycle":"Bike"}
REAL_HEIGHT       = {"person":1.7,"car":1.5,"truck":3.0,
                     "bus":3.2,"motorcycle":1.4,"bicycle":1.4}
FOCAL_LENGTH      = 280
BLIND_RATIO       = 0.2
ALERT_DISTANCE    = 5.0
CRITICAL_DIST     = 3.0
COLLISION_STOP_FRAMES = 30

# GPIO pins (BCM)
GPIO_BUTTON_PIN     = 5
GPIO_BUZZER_PIN     = 12
GPIO_LED_MEDIUM_PIN = 17
GPIO_LED_HIGH_PIN   = 27
GPIO_LEFT_LED_PIN   = 23
GPIO_RIGHT_LED_PIN  = 24
GPIO_FIT_LED_PIN    = 25
BUZZER_SHORT_MS     = 200
BUZZER_LONG_MS      = 800

# MediaPipe landmark indices
L_EYE    = [33,  160, 158, 133, 153, 144]
R_EYE    = [362, 385, 387, 263, 373, 380]
M_TOP    = 13;  M_BOT   = 14
M_LEFT   = 78;  M_RIGHT = 308
NOSE_TIP = 1

# ─────────────────────────────────────────────────────────────────
# SYSTEM STATE MACHINE
# ─────────────────────────────────────────────────────────────────
SYSTEM_MODE = "FIT_TEST"
_mode_lock  = threading.Lock()

def get_mode():
    with _mode_lock:
        return SYSTEM_MODE

def set_mode(mode):
    global SYSTEM_MODE
    with _mode_lock:
        SYSTEM_MODE = mode
    try:
        socketio.emit("system_mode", {"mode": mode})
    except Exception:
        pass
    print(f"[System] Mode → {mode}")

# ─────────────────────────────────────────────────────────────────
# FLASK / SOCKETIO
# ─────────────────────────────────────────────────────────────────
app      = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─────────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────────
state = {
    # Driver
    "ear":0.0,"mar":0.0,"eye_status":"Unknown",
    "yawn":False,"head_nod":False,"hand_near_mouth":False,
    "driver_present":False,"driver_status":"Initializing",
    "drowsiness_events":0,"yawn_events":0,"head_nod_events":0,
    "combined_events":0,"landmarks":[],
    # Road
    "objects":[],"collision_risk":"LOW","risk_score":0.0,
    "closest_distance":None,"closest_object":"None",
    "ttc":None,"blind_spot":"CLEAR","object_count":0,
    # System
    "driver_fps":0.0,"road_fps":0.0,
    "uptime":0,"road_cam_active":False,
    "stream_w":STREAM_WIDTH,"stream_h":STREAM_HEIGHT,
    "system_mode":"FIT_TEST",
    # Drunk
    "drunk_score":0.0,"drunk_detected":False,"drunk_warning":False,
    "sensor_active":False,"sensor_reading":0.0,
    # Relay
    "relay_locked":True,"relay_reason":"Awaiting fit test",
    "relay_countdown":0,"relay_phase":"idle",
    # Fit test
    "fit_test_status":"idle","fit_test_phase":None,
    "fit_test_phase_detail":"","fit_test_score":0.0,
    "fit_test_last_ts":None,"fit_test_last_result":None,
    "fit_test_phases":{},"fit_test_math_question":None,
}
state_lock = threading.Lock()
start_time = time.time()

driver_frame_bytes = None
road_frame_bytes   = None
frame_lock_driver  = threading.Lock()
frame_lock_road    = threading.Lock()

driver_queue = Queue(maxsize=2)
road_queue   = Queue(maxsize=2)

road_cam_stop_event     = threading.Event()
road_cam_stop_event.set()   # starts inactive

fit_test_button_event = threading.Event()

# Notification dedup flags (in-memory, complement to notifier's own dedup)
_drunk_notif_fired    = False
_drunk_notif_lock     = threading.Lock()
_fatigue_notif_fired  = False

# ─────────────────────────────────────────────────────────────────
# GPIO
# ─────────────────────────────────────────────────────────────────
_gpio_button  = None
_gpio_buzzer  = None
_led_medium   = None
_led_high     = None
_left_led     = None
_right_led    = None
_fit_led      = None


def setup_gpio():
    global _gpio_button,_gpio_buzzer,_led_medium,_led_high
    global _left_led,_right_led,_fit_led
    if not gpio_available:
        print("[GPIO] Skipped")
        return
    try:
        Device.pin_factory = LGPIOFactory(chip=0)
        _gpio_button = Button(GPIO_BUTTON_PIN, pull_up=True, bounce_time=0.3)

        def _on_press():
            mode = get_mode()
            if mode == "FIT_TEST":
                fit_test_button_event.set()
            else:
                relay.cancel_escalation()

        _gpio_button.when_pressed = _on_press

        _gpio_buzzer = Buzzer(GPIO_BUZZER_PIN);  _gpio_buzzer.off()
        _led_medium  = LED(GPIO_LED_MEDIUM_PIN); _led_medium.off()
        _led_high    = LED(GPIO_LED_HIGH_PIN);   _led_high.off()
        _left_led    = LED(GPIO_LEFT_LED_PIN);   _left_led.off()
        _right_led   = LED(GPIO_RIGHT_LED_PIN);  _right_led.off()
        _fit_led     = LED(GPIO_FIT_LED_PIN);    _fit_led.off()
        print("[GPIO] All pins ready")
    except Exception as e:
        print(f"[GPIO] Setup failed: {e}")


def buzzer_beep(duration_ms: int):
    if not gpio_available or _gpio_buzzer is None:
        return
    def _pulse():
        _gpio_buzzer.on()
        time.sleep(duration_ms / 1000.0)
        _gpio_buzzer.off()
    threading.Thread(target=_pulse, daemon=True).start()


def fit_led_fn(val: bool):
    if gpio_available and _fit_led:
        try:
            _fit_led.on() if val else _fit_led.off()
        except Exception:
            pass


def update_risk_leds(risk: str):
    if not gpio_available or _led_medium is None:
        return
    if risk == "HIGH":
        _led_medium.on(); _led_high.on()
    elif risk == "MEDIUM":
        _led_medium.on(); _led_high.off()
    else:
        _led_medium.off(); _led_high.off()


def hazard_on(speed=0.5):
    if not gpio_available or _left_led is None:
        return
    _left_led.blink(on_time=speed, off_time=speed, background=True)
    _right_led.blink(on_time=speed, off_time=speed, background=True)


def hazard_off():
    if not gpio_available or _left_led is None:
        return
    _left_led.off(); _right_led.off()


def update_blind_leds(status: str):
    if not gpio_available or _left_led is None:
        return
    if status == "LEFT":
        _left_led.on(); _right_led.off()
    elif status == "RIGHT":
        _right_led.on(); _left_led.off()
    else:
        _left_led.off(); _right_led.off()

# ─────────────────────────────────────────────────────────────────
# FIT TEST ORCHESTRATION
# ─────────────────────────────────────────────────────────────────
def _on_fit_test_complete(result, test_state):
    score = test_state.get("score", 0.0)
    if result == "passed":
        relay.unlock("fit_test_passed")
        notif.reset_session()
        dd.reset()
        set_mode("MONITORING")
        socketio.emit("fit_test_result", {"result":"passed","state":test_state})
        notif.notify_fit_test_pass(DRIVER_ID, score,
                                   uptime_s=time.time()-start_time)
        print("[FitTest] PASSED — monitoring active")
    else:
        reason = test_state.get("fail_reason","Unknown")
        relay.lock_immediate(f"Fit test failed: {reason}")
        set_mode("LOCKED")
        socketio.emit("fit_test_result",
                      {"result":"failed","state":test_state,"reason":reason})
        notif.notify_fit_test_fail(DRIVER_ID, reason, score,
                                   uptime_s=time.time()-start_time)
        buzzer_beep(BUZZER_LONG_MS * 2)
        print(f"[FitTest] FAILED — {reason}")


def launch_fit_test():
    global _drunk_notif_fired, _fatigue_notif_fired
    set_mode("FIT_TEST")
    relay.lock_immediate("Fit test in progress")
    _drunk_notif_fired   = False
    _fatigue_notif_fired = False

    threading.Thread(
        target=ft.run_fit_test,
        kwargs=dict(
            shared_state=state,
            state_lock=state_lock,
            button_pressed_event=fit_test_button_event if gpio_available else None,
            buzzer_fn=buzzer_beep,
            led_fn=fit_led_fn,
            on_complete_cb=_on_fit_test_complete,
            socketio=socketio,
        ),
        daemon=True,
    ).start()
    print("[FitTest] Launched")

# ─────────────────────────────────────────────────────────────────
# DRUNK ALERT
# ─────────────────────────────────────────────────────────────────
def _check_drunk(drunk_score, drunk_detected, uptime_s):
    global _drunk_notif_fired
    if not drunk_detected:
        return
    with _drunk_notif_lock:
        if _drunk_notif_fired:
            return
        _drunk_notif_fired = True

    print(f"[Drunk] DETECTED score={drunk_score:.0%} — alerting")
    notif.notify_drunk_detected(DRIVER_ID, drunk_score, uptime_s)
    buzzer_beep(BUZZER_LONG_MS)
    socketio.emit("drunk_alert", {
        "score":   drunk_score,
        "message": "Impairment detected — driver alerted.",
    })

# ─────────────────────────────────────────────────────────────────
# ROAD CAM GATE
# ─────────────────────────────────────────────────────────────────
def start_road_cam():
    if road_cam_stop_event.is_set():
        road_cam_stop_event.clear()
        with state_lock:
            state["road_cam_active"] = True
        socketio.emit("road_cam_started", {})
        print("[Road] Activated by fatigue trigger")


def stop_road_cam():
    if not road_cam_stop_event.is_set():
        road_cam_stop_event.set()
        with state_lock:
            state.update({
                "road_cam_active":False,"objects":[],"collision_risk":"LOW",
                "risk_score":0.0,"closest_distance":None,
                "closest_object":"None","ttc":None,
                "blind_spot":"CLEAR","object_count":0,
            })
        update_risk_leds("LOW")
        hazard_off()
        socketio.emit("road_cam_stopped", {})
        print("[Road] Stopped")

# ─────────────────────────────────────────────────────────────────
# FATIGUE HELPERS
# ─────────────────────────────────────────────────────────────────
def _dist(p1, p2):
    return hypot(p1[0]-p2[0], p1[1]-p2[1])

def compute_ear(lm, idx):
    p = [lm[i] for i in idx]
    return (_dist(p[1],p[5]) + _dist(p[2],p[4])) / (2.0*_dist(p[0],p[3]) + 1e-6)

def compute_mar(lm):
    return _dist(lm[M_TOP],lm[M_BOT]) / (_dist(lm[M_LEFT],lm[M_RIGHT]) + 1e-6)

def is_nodding(lm, fh):
    return lm[NOSE_TIP][1] > fh * HEAD_DROP_THRESHOLD

def is_hand_near_mouth(hand_res, face_lm, w, h, mar):
    if not hand_res.multi_hand_landmarks or mar < 0.20:
        return False
    pad = 30
    mx  = [face_lm[i][0] for i in [M_TOP,M_BOT,M_LEFT,M_RIGHT]]
    my  = [face_lm[i][1] for i in [M_TOP,M_BOT,M_LEFT,M_RIGHT]]
    x1,x2 = min(mx)-pad, max(mx)+pad
    y1,y2 = min(my)-pad, max(my)+pad
    for hand in hand_res.multi_hand_landmarks:
        for tip in [4,8,12,16,20]:
            lm2 = hand.landmark[tip]
            hx,hy = int(lm2.x*w), int(lm2.y*h)
            if x1<=hx<=x2 and y1<=hy<=y2:
                return True
    return False

# ─────────────────────────────────────────────────────────────────
# COLLISION HELPERS
# ─────────────────────────────────────────────────────────────────
def est_distance(height_px, label):
    rh = REAL_HEIGHT.get(label, 1.7)
    return round((rh * FOCAL_LENGTH) / height_px, 2) if height_px else 999.0

def get_centre(x1,y1,x2,y2):
    return ((x1+x2)//2, (y1+y2)//2)

def calc_ttc(prev, curr, dt):
    if prev is None or dt <= 0:
        return None
    speed = (prev-curr)/dt
    if speed <= 0:
        return None
    ttc = curr/speed
    return round(ttc,2) if 0 < ttc <= 20 else None

def blind_zone(cx, fw):
    left  = int(fw*BLIND_RATIO)
    right = int(fw*(1-BLIND_RATIO))
    if cx < left:  return "LEFT"
    if cx > right: return "RIGHT"
    return "CENTER"

# ─────────────────────────────────────────────────────────────────
# DRIVER CAPTURE WORKER
# ─────────────────────────────────────────────────────────────────
def driver_capture_worker():
    picam2 = Picamera2(0)
    picam2.configure(picam2.create_preview_configuration(
        main={"size":(STREAM_WIDTH,STREAM_HEIGHT),"format":"BGR888"}
    ))
    picam2.start()
    time.sleep(2)
    print("[Driver] CSI camera started")
    while True:
        frame = picam2.capture_array()
        if not driver_queue.full():
            driver_queue.put(frame)

# ─────────────────────────────────────────────────────────────────
# DRIVER AI WORKER
# ─────────────────────────────────────────────────────────────────
def driver_ai_worker():
    global driver_frame_bytes, _fatigue_notif_fired

    if not mp_available:
        print("[Driver] MediaPipe unavailable — skipped")
        return
    if not os.path.exists(FACE_TASK_PATH):
        print(f"[Driver] ERROR: {FACE_TASK_PATH} not found")
        return

    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_TASK_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.4,
    )
    landmarker  = FaceLandmarker.create_from_options(opts)
    mp_hands    = mp.solutions.hands
    hands_model = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
    print("[Driver] MediaPipe loaded")

    eye_ctr=yawn_ctr=nod_ctr=0
    d_evt=y_evt=n_evt=0
    prev_mar=0.0; blink_cd=0; prev_combined=0
    fps_ctr=0; fps_t=time.time()
    t_start=time.monotonic()

    while True:
        if driver_queue.empty():
            time.sleep(0.005)
            continue

        frame  = driver_queue.get()
        fh, fw = frame.shape[:2]
        monitoring = (get_mode() == "MONITORING")

        if blink_cd > 0:
            blink_cd -= 1

        EAR=MAR=0.0
        present=yawn_det=nod_det=hand_near=False
        eye_st="Unknown"; pts_out=[]; d_status="Alert"; nose_x=fw//2

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int((time.monotonic()-t_start)*1000)
        face_res = landmarker.detect_for_video(mp_img, ts_ms)
        hand_res = hands_model.process(rgb)

        if face_res.face_landmarks:
            present = True
            face    = face_res.face_landmarks[0]
            lm      = [(int(p.x*fw), int(p.y*fh)) for p in face]
            pts_out = [[x,y] for x,y in lm]
            nose_x  = lm[NOSE_TIP][0]

            EAR = (compute_ear(lm,L_EYE) + compute_ear(lm,R_EYE)) / 2.0
            MAR = compute_mar(lm)

            if EAR < EAR_THRESHOLD:
                if blink_cd == 0: eye_ctr += 1
                eye_st = "Closed"
            else:
                if 0 < eye_ctr <= 6: blink_cd = 10
                eye_ctr = 0; eye_st = "Open"

            if monitoring and eye_ctr > EYE_FRAMES and d_evt < COMBINED_EVENT_LIMIT:
                d_evt += 1; eye_ctr = 0; d_status = "Drowsy"

            hand_near  = is_hand_near_mouth(hand_res, lm, fw, fh, MAR)
            yawn_sig   = (MAR > MAR_THRESHOLD
                          or (hand_near and MAR > 0.20)
                          or (MAR < 0.1 and prev_mar > MAR_THRESHOLD))
            if yawn_sig: yawn_ctr += 1; yawn_det = True
            else:        yawn_ctr = 0

            if monitoring and yawn_ctr > YAWN_FRAMES and y_evt < COMBINED_EVENT_LIMIT:
                y_evt += 1; yawn_ctr = 0; d_status = "Yawning"

            if is_nodding(lm, fh): nod_ctr += 1; nod_det = True
            else:                  nod_ctr = 0

            if monitoring and nod_ctr > HEAD_NOD_FRAMES and n_evt < COMBINED_EVENT_LIMIT:
                n_evt += 1; nod_ctr = 0; d_status = "Head Nod"

            prev_mar = MAR
        else:
            d_status = "Face Not Detected"

        combined = min(d_evt + y_evt + n_evt, COMBINED_EVENT_LIMIT)
        if combined >= COMBINED_EVENT_LIMIT:
            d_status = "Critical"

        # ── Drunk detection ────────────────────────────────────────────
        if monitoring:
            dd.update({
                "ear":EAR,"mar":MAR,"head_nod":nod_det,
                "eye_status":eye_st,"driver_present":present,
                "nose_x":nose_x,
            })
            di = dd.get_state()
            drunk_score    = di["drunk_score"]
            drunk_detected = di["drunk_detected"]
            drunk_warning  = di["drunk_warning"]
            _check_drunk(drunk_score, drunk_detected, time.time()-start_time)
        else:
            di = dd.get_state()
            drunk_score = di["drunk_score"]
            drunk_detected = di["drunk_detected"]
            drunk_warning  = di["drunk_warning"]

        # ── Fatigue notifications ──────────────────────────────────────
        if monitoring and combined >= COMBINED_EVENT_LIMIT and not _fatigue_notif_fired:
            _fatigue_notif_fired = True
            notif.notify_critical_fatigue(DRIVER_ID, combined,
                                          uptime_s=time.time()-start_time)
            buzzer_beep(BUZZER_LONG_MS)
            socketio.emit("fatigue_alert", {
                "events":  combined,
                "message": "Critical fatigue — driver & supervisor alerted.",
            })
            print(f"[Fatigue] CRITICAL {combined}/5 — alerting")
        elif monitoring and combined >= COMBINED_WARN_LEVEL and combined > prev_combined:
            buzzer_beep(BUZZER_SHORT_MS)

        # ── Road cam gate ──────────────────────────────────────────────
        if monitoring and combined >= ROAD_CAM_TRIGGER:
            start_road_cam()

        # ── Driver cam overlays ────────────────────────────────────────
        if monitoring and drunk_detected:
            cv2.putText(frame, "IMPAIRMENT DETECTED",
                (10,fh-30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
        elif monitoring and drunk_warning:
            cv2.putText(frame, f"WARNING {drunk_score:.0%}",
                (10,fh-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,165,255), 1)

        # FPS
        fps_ctr += 1
        if time.time()-fps_t >= 1.0:
            with state_lock:
                state["driver_fps"] = round(fps_ctr/(time.time()-fps_t), 1)
            fps_ctr=0; fps_t=time.time()

        prev_combined = combined

        # Relay + fit test snapshots
        rs  = relay.get_state()
        fts = ft.fit_test_state

        with state_lock:
            state.update({
                "ear":round(EAR,3),"mar":round(MAR,3),
                "eye_status":eye_st,"yawn":yawn_det,"head_nod":nod_det,
                "hand_near_mouth":hand_near,"driver_present":present,
                "driver_status":d_status,
                "drowsiness_events":d_evt,"yawn_events":y_evt,
                "head_nod_events":n_evt,"combined_events":combined,
                "landmarks":pts_out,
                "system_mode":get_mode(),
                "drunk_score":   round(drunk_score,3),
                "drunk_detected":bool(drunk_detected),
                "drunk_warning": bool(drunk_warning),
                "sensor_active": di.get("sensor_active",False),
                "sensor_reading":di.get("sensor_reading",0.0),
                "relay_locked":   rs["locked"],
                "relay_reason":   rs["lock_reason"],
                "relay_countdown":rs["countdown_s"],
                "relay_phase":    rs["escalation_phase"],
                "fit_test_status":      fts["status"],
                "fit_test_phase":       fts["phase"],
                "fit_test_phase_detail":fts["phase_detail"],
                "fit_test_score":       fts["score"],
                "fit_test_last_ts":     fts["last_run_ts"],
                "fit_test_last_result": fts["last_result"],
                "fit_test_phases":      fts.get("phases",{}),
                "fit_test_math_question": fts.get("math_question"),
            })

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock_driver:
            driver_frame_bytes = buf.tobytes()

# ─────────────────────────────────────────────────────────────────
# ROAD CAPTURE WORKER
# ─────────────────────────────────────────────────────────────────
def road_capture_worker():
    time.sleep(4)
    cap = None
    for i in range(5):
        if i == 0: continue
        c = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not c.isOpened(): c.release(); continue
        ret, f = c.read()
        if ret and f is not None:
            cap = c
            print(f"[Road] USB camera at /dev/video{i}")
            break
        c.release()
    if cap is None:
        print("[Road] No USB camera found"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          STREAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    while True:
        if road_cam_stop_event.is_set():
            time.sleep(0.2); continue
        cap.grab()
        ret, frame = cap.retrieve()
        if ret and frame is not None and not road_queue.full():
            road_queue.put(frame)

# ─────────────────────────────────────────────────────────────────
# ROAD AI WORKER
# ─────────────────────────────────────────────────────────────────
def road_ai_worker():
    global road_frame_bytes
    model = YOLO(YOLO_MODEL_PATH) if yolo_available else None
    if model: print("[Road] YOLO loaded")

    prev_dist = None; prev_t = time.time()
    frame_skip = 0; yolo_res = None
    high_streak = 0; prev_risk = "LOW"
    fps_ctr = 0; fps_t = time.time()

    while True:
        if road_queue.empty():
            time.sleep(0.005); continue
        if road_cam_stop_event.is_set():
            road_queue.get(); continue

        frame = road_queue.get()
        h, w  = frame.shape[:2]
        now = time.time(); dt = now-prev_t; prev_t = now

        frame_skip += 1
        if frame_skip % 4 == 0 and model:
            yolo_res = model(frame, conf=0.45, imgsz=256, verbose=False)

        closest=999.0; closest_obj="None"
        objects_out=[]; blind_st="CLEAR"; count=0

        if yolo_res:
            for r in yolo_res:
                for box in r.boxes:
                    cls   = int(box.cls[0])
                    label = model.names[cls]
                    if label not in IMPORTANT_OBJECTS: continue
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    dist  = est_distance(y2-y1, label)
                    cx,cy = get_centre(x1,y1,x2,y2)
                    conf  = float(box.conf[0])
                    spot  = blind_zone(cx, w)
                    if dist < closest: closest=dist; closest_obj=OBJECT_DISPLAY.get(label,label)
                    if spot in ("LEFT","RIGHT"): blind_st=spot
                    col=(0,255,0) if dist>=ALERT_DISTANCE else (0,165,255) if dist>=CRITICAL_DIST else (0,0,255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),col,1)
                    cv2.putText(frame,f"{OBJECT_DISPLAY.get(label,label)} {dist:.1f}m",
                                (x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1)
                    objects_out.append({
                        "label":OBJECT_DISPLAY.get(label,label),
                        "conf":round(conf,2),"distance":dist,
                        "blind_spot":spot,"box":[x1,y1,x2,y2],
                        "frame_w":w,"frame_h":h,
                    })
                    count+=1

        ttc_val   = calc_ttc(prev_dist, closest, dt)
        prev_dist = closest

        risk_str  = "HIGH" if closest<CRITICAL_DIST else "MEDIUM" if closest<ALERT_DISTANCE else "LOW"
        risk_score= 1.0   if risk_str=="HIGH"       else 0.5      if risk_str=="MEDIUM"      else 0.1

        if closest<CRITICAL_DIST: hazard_on(0.15)
        elif closest<ALERT_DISTANCE: hazard_on(0.5)
        else: hazard_off(); update_blind_leds(blind_st)
        update_risk_leds(risk_str)

        if risk_str!="LOW" and risk_str!=prev_risk:
            socketio.emit("collision_warning",
                          {"risk":risk_str,"risk_score":round(risk_score,2),
                           "objects":objects_out})
        prev_risk = risk_str

        if risk_str=="HIGH": high_streak+=1
        else: high_streak=0
        if high_streak >= COLLISION_STOP_FRAMES:
            socketio.emit("collision_alert",
                          {"risk":risk_str,"objects":objects_out,
                           "message":"Sustained HIGH risk — road feed paused"})
            stop_road_cam(); high_streak=0; continue

        fps_ctr+=1
        if time.time()-fps_t>=1.0:
            with state_lock:
                state["road_fps"]=round(fps_ctr/(time.time()-fps_t),1)
            fps_ctr=0; fps_t=time.time()

        with state_lock:
            state.update({
                "objects":objects_out,"collision_risk":risk_str,
                "risk_score":round(risk_score,2),
                "closest_distance":None if closest>=999 else closest,
                "closest_object":closest_obj,"ttc":ttc_val,
                "blind_spot":blind_st,"object_count":count,
            })

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock_road:
            road_frame_bytes = buf.tobytes()

# ─────────────────────────────────────────────────────────────────
# STREAM GENERATORS
# ─────────────────────────────────────────────────────────────────
def gen_driver():
    while True:
        with frame_lock_driver:
            f = driver_frame_bytes
        if f:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + f + b'\r\n'
        time.sleep(1/STREAM_FPS)

def gen_road():
    while True:
        if road_cam_stop_event.is_set():
            time.sleep(0.2); continue
        with frame_lock_road:
            f = road_frame_bytes
        if f:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + f + b'\r\n'
        time.sleep(1/STREAM_FPS)

# ─────────────────────────────────────────────────────────────────
# TELEMETRY — 10 Hz
# ─────────────────────────────────────────────────────────────────
def telemetry_worker():
    while True:
        with state_lock:
            payload = dict(state)
        payload["uptime"]      = round(time.time()-start_time)
        payload["system_mode"] = get_mode()
        rs = relay.get_state()
        payload.update({
            "relay_locked":   rs["locked"],
            "relay_phase":    rs["escalation_phase"],
            "relay_countdown":rs["countdown_s"],
            "relay_reason":   rs["lock_reason"],
        })
        socketio.emit("telemetry", payload)
        time.sleep(0.1)

# ─────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────
@app.route("/stream/driver")
def stream_driver():
    return Response(gen_driver(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stream/road")
def stream_road():
    return Response(gen_road(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/health")
def health():
    return jsonify({"status":"ok","uptime":round(time.time()-start_time),
                    "mode":get_mode()})

@app.route("/stop_road", methods=["POST"])
def api_stop_road():
    stop_road_cam()
    return jsonify({"road_cam_active":False})

@app.route("/start_road", methods=["POST"])
def api_start_road():
    start_road_cam()
    return jsonify({"road_cam_active":True})

@app.route("/fit_test/start", methods=["POST"])
def api_fit_test_start():
    if get_mode() == "FIT_TEST" and ft.fit_test_state["status"] == "running":
        return jsonify({"error":"already running"}), 409
    launch_fit_test()
    return jsonify({"status":"started"})

@app.route("/fit_test/retry", methods=["POST"])
def api_fit_test_retry():
    launch_fit_test()
    return jsonify({"status":"restarted"})

@app.route("/fit_test/result")
def api_fit_test_result():
    with ft.fit_test_lock:
        return jsonify(dict(ft.fit_test_state))

@app.route("/relay/unlock", methods=["POST"])
def api_relay_unlock():
    data   = request.get_json(silent=True) or {}
    reason = data.get("reason","operator_override")
    relay.unlock(reason)
    set_mode("MONITORING")
    return jsonify({"locked":False,"reason":reason})

@app.route("/relay/lock", methods=["POST"])
def api_relay_lock():
    data   = request.get_json(silent=True) or {}
    reason = data.get("reason","manual_lock")
    relay.lock_immediate(reason)
    set_mode("LOCKED")
    return jsonify({"locked":True,"reason":reason})

@app.route("/notifications")
def api_notifications():
    return jsonify(notif.get_log())

@app.route("/drunk_state")
def api_drunk_state():
    return jsonify(dd.get_state())

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_gpio()
    relay.setup(socketio=socketio)

    print("="*56)
    print("  Edge AI Driver Safety — Pi 5  (v2)")
    print(f"  Port      : {PORT}")
    print(f"  MediaPipe : {'enabled' if mp_available else 'DISABLED'}")
    print(f"  YOLO      : {'enabled' if yolo_available else 'DISABLED'}")
    print(f"  GPIO      : {'enabled' if gpio_available else 'DISABLED'}")
    print(f"  Dry-run   : {notif.CFG['DRY_RUN']}")
    print("="*56)

    relay.lock_immediate("System startup — awaiting fit test")
    set_mode("FIT_TEST")

    for target in [driver_capture_worker, driver_ai_worker,
                   road_capture_worker,   road_ai_worker,
                   telemetry_worker]:
        threading.Thread(target=target, daemon=True).start()

    def _auto_fit_test():
        time.sleep(3)
        launch_fit_test()
    threading.Thread(target=_auto_fit_test, daemon=True).start()

    socketio.run(app, host="0.0.0.0", port=PORT, debug=False)
