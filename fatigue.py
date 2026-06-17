import cv2
import mediapipe as mp
import numpy as np
import time
import os
from math import hypot

# -------------------------------------------------
# Mediapipe imports
# -------------------------------------------------

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode
)

import mediapipe.python.solutions.hands as mp_hands

hands_model = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6
)

# -------------------------------------------------
# Thresholds
# -------------------------------------------------

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.35

EYE_FRAMES = 20
YAWN_FRAMES = 12
HEAD_NOD_FRAMES = 10

COMBINED_EVENT_LIMIT = 5
HEAD_DROP_THRESHOLD = 0.65

# -------------------------------------------------
# Counters
# -------------------------------------------------

eye_counter = 0
yawn_counter = 0
head_nod_counter = 0

drowsiness_events = 0
yawn_events = 0
head_nod_events = 0

prev_mar = 0.0
blink_cooldown = 0
yawn_cooldown = 0

# -------------------------------------------------
# Landmark indices
# -------------------------------------------------

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308
NOSE_TIP = 1

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def distance(p1,p2):
    return hypot(p1[0]-p2[0],p1[1]-p2[1])

def compute_ear(lm,eye):

    p1,p2,p3,p4,p5,p6=[lm[i] for i in eye]

    v1=distance(p2,p6)
    v2=distance(p3,p5)
    h=distance(p1,p4)

    return (v1+v2)/(2.0*h)

def compute_mar(lm):

    top=lm[UPPER_LIP]
    bot=lm[LOWER_LIP]
    left=lm[LEFT_MOUTH]
    right=lm[RIGHT_MOUTH]

    v=distance(top,bot)
    h=distance(left,right)

    return v/h

def is_head_nodding(lm,h):

    nose_y=lm[NOSE_TIP][1]
    return nose_y>(h*HEAD_DROP_THRESHOLD)

def is_hand_near_mouth(hand_res,face_lm,w,h,mar):

    if not hand_res.multi_hand_landmarks:
        return False

    if mar<0.20:
        return False

    idx=[UPPER_LIP,LOWER_LIP,LEFT_MOUTH,RIGHT_MOUTH]

    mx=[face_lm[i][0] for i in idx]
    my=[face_lm[i][1] for i in idx]

    pad=30

    x1,x2=min(mx)-pad,max(mx)+pad
    y1,y2=min(my)-pad,max(my)+pad

    fingertips=[4,8,12,16,20]

    for hand in hand_res.multi_hand_landmarks:

        for tip in fingertips:

            lm=hand.landmark[tip]

            hx=int(lm.x*w)
            hy=int(lm.y*h)

            if x1<=hx<=x2 and y1<=hy<=y2:
                return True

    return False

# -------------------------------------------------
# UI functions
# -------------------------------------------------

def draw_event_bar(frame,label,count,limit,x,y,color,w=280,h=28):

    capped=min(count,limit)
    filled=int((capped/limit)*w)

    cv2.rectangle(frame,(x+3,y+3),(x+w+3,y+h+3),(0,0,0),-1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(30,30,30),-1)

    if filled>0:
        cv2.rectangle(frame,(x,y),(x+filled,y+h),color,-1)

    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

    txt=f"{label}: {capped}/{limit}"

    cv2.putText(frame,txt,(x,y-6),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2)

def draw_hud(frame,EAR,MAR,hand_near,status,color,h,w):

    overlay=frame.copy()

    cv2.rectangle(overlay,(10,10),(280,105),(0,0,0),-1)

    cv2.addWeighted(overlay,0.5,frame,0.5,0,frame)

    cv2.putText(frame,f"EAR : {EAR:.3f}",(20,45),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,100),2)

    cv2.putText(frame,f"MAR : {MAR:.3f}",(20,90),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,255),2)

    if hand_near:

        cv2.putText(frame,"Hand near mouth",(20,130),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,215,255),2)

    scale=1.1
    thick=3

    (tw,th),_=cv2.getTextSize(status,
                              cv2.FONT_HERSHEY_SIMPLEX,
                              scale,
                              thick)

    bx=(w-tw)//2
    by=160

    overlay2=frame.copy()

    cv2.rectangle(overlay2,(bx-15,by-th-10),(bx+tw+15,by+12),(0,0,0),-1)

    cv2.addWeighted(overlay2,0.6,frame,0.4,0,frame)

    cv2.putText(frame,status,(bx,by),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,color,thick)

# -------------------------------------------------
# Load FaceLandmarker
# -------------------------------------------------

MODEL_PATH=os.path.join(os.path.dirname(__file__),"face_landmarker.task")

options=FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

landmarker=FaceLandmarker.create_from_options(options)

print("FaceLandmarker loaded")

# -------------------------------------------------
# Webcam
# -------------------------------------------------

cap=cv2.VideoCapture(0)

start=time.time()

while True:

    ret,frame=cap.read()

    if not ret:
        break

    frame=cv2.flip(frame,1)

    h,w,_=frame.shape

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_img=mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp=int((time.time()-start)*1000)

    result=landmarker.detect_for_video(mp_img,timestamp)

    hand_res=hands_model.process(rgb)

 

    if blink_cooldown>0:
        blink_cooldown-=1

    if yawn_cooldown>0:
        yawn_cooldown-=1

    combined=min(drowsiness_events+yawn_events+head_nod_events,
                 COMBINED_EVENT_LIMIT)

    status="Driver Alert"
    status_color=(0,220,0)

    if combined>=COMBINED_EVENT_LIMIT:
        status="CRITICAL FATIGUE - STOP DRIVING"
        status_color=(0,0,255)

    EAR=0
    MAR=0
    hand_near=False

    if result.face_landmarks:

        face=result.face_landmarks[0]

        lm=[(int(p.x*w),int(p.y*h)) for p in face]

        EAR=(compute_ear(lm,LEFT_EYE)+compute_ear(lm,RIGHT_EYE))/2

        MAR=compute_mar(lm)

        if EAR<EAR_THRESHOLD:

            if blink_cooldown==0:
                eye_counter+=1

        else:

            if 0<eye_counter<=6:
                blink_cooldown=10

            eye_counter=0

        if eye_counter>EYE_FRAMES and drowsiness_events<COMBINED_EVENT_LIMIT:

            drowsiness_events+=1
            eye_counter=0

            status="DROWSINESS DETECTED"
            status_color=(0,165,255)

        hand_near=is_hand_near_mouth(hand_res,lm,w,h,MAR)

        yawn_signal=(MAR>MAR_THRESHOLD
                     or (hand_near and MAR>0.25))

        if yawn_signal:
            yawn_counter+=1
        else:
            yawn_counter=0

        if yawn_counter>YAWN_FRAMES and yawn_cooldown==0:

            if yawn_events<COMBINED_EVENT_LIMIT:

                yawn_events+=1
                yawn_counter=0
                yawn_cooldown=30

                status="YAWNING DETECTED"
                status_color=(0,255,255)

        if is_head_nodding(lm,h):
            head_nod_counter+=1
        else:
            head_nod_counter=0

        if head_nod_counter>HEAD_NOD_FRAMES:

            if head_nod_events<COMBINED_EVENT_LIMIT:

                head_nod_events+=1
                head_nod_counter=0

                status="HEAD NOD DETECTED"
                status_color=(0,80,255)

        for (x,y) in lm:
            cv2.circle(frame,(x,y),1,(0,255,0),-1)

    draw_hud(frame,EAR,MAR,hand_near,status,status_color,h,w)

    x=20

    draw_event_bar(frame,"Drowsy Events",drowsiness_events,
                   COMBINED_EVENT_LIMIT,x,h-210,(0,165,255))

    draw_event_bar(frame,"Yawn Events",yawn_events,
                   COMBINED_EVENT_LIMIT,x,h-155,(0,255,255))

    draw_event_bar(frame,"Head Nod Events",head_nod_events,
                   COMBINED_EVENT_LIMIT,x,h-100,(0,80,255))

    draw_event_bar(frame,"Combined",combined,
                   COMBINED_EVENT_LIMIT,x,h-45,(0,0,255))

    cv2.imshow("Driver Fatigue Monitoring",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()