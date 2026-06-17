"""
fit_test.py — Pre-Drive Fitness Test (6 Phases)
================================================
Phases (in order):
  1. face_presence   — face must be detected and stable for 3 seconds
  2. eye_openness    — EAR must be above baseline threshold for 3 seconds
  3. head_position   — nose tip must be centred, pitch/yaw within limits
  4. gaze_tracking   — follow a moving dot: LEFT → CENTRE → RIGHT
                       tracked via iris landmarks (MediaPipe 478-pt model)
  5. reaction_time   — LED blinks 3 times randomly; driver presses button
                       within 1.5 s each time; average RT scored
  6. math_question   — simple arithmetic shown on dashboard;
                       driver presses GPIO button: short=A, double=B, long=C

Pass threshold: 5 out of 6 phases (one phase may be skipped/failed).
Composite score = mean of individual phase scores.

Each phase updates shared state so the dashboard shows live progress.
on_complete_cb(result, state) is called when the test finishes.
"""

import time
import threading
import random

# ── Phase config ────────────────────────────────────────────────────────────
FACE_STABLE_SECS    = 3.0    # face must be present this long
EAR_OPEN_THRESHOLD  = 0.22   # below this = eyes not open enough
EAR_STABLE_SECS     = 3.0
HEAD_CENTRE_TOL     = 0.18   # nose tip must be in centre 18% of frame width
HEAD_PITCH_TOL      = 0.20   # nose tip Y must not be too high/low
HEAD_STABLE_SECS    = 2.5
GAZE_HOLD_SECS      = 1.2    # must hold each gaze direction this long
GAZE_TIMEOUT_SECS   = 8.0
REACTION_ROUNDS     = 3
REACTION_TIMEOUT    = 1.5    # seconds to press after LED
MATH_TIMEOUT        = 15.0   # seconds to answer math question
PHASE_TIMEOUT       = 15.0   # max seconds any phase may take

# Scoring weights (must sum to 1.0)
PHASE_WEIGHTS = {
    "face_presence":  0.10,
    "eye_openness":   0.20,
    "head_position":  0.15,
    "gaze_tracking":  0.20,
    "reaction_time":  0.20,
    "math_question":  0.15,
}
PASS_SCORE = 0.65   # composite score to pass overall

# ── Shared test state (read by app.py for telemetry) ────────────────────────
fit_test_lock  = threading.Lock()
fit_test_state = {
    "status":       "idle",     # idle | running | passed | failed
    "phase":        None,
    "phase_detail": "",
    "score":        0.0,
    "phases":       {},         # {phase_name: {pass, score, detail}}
    "fail_reason":  "",
    "last_result":  None,
    "last_run_ts":  None,
}

# ── Math question bank ──────────────────────────────────────────────────────
_MATH_QUESTIONS = [
    {"q": "7 + 5 = ?",  "a": "12", "options": ["10", "12", "14"], "answer_idx": 1},
    {"q": "9 × 3 = ?",  "a": "27", "options": ["24", "27", "30"], "answer_idx": 1},
    {"q": "15 - 8 = ?", "a": "7",  "options": ["5",  "7",  "9"],  "answer_idx": 1},
    {"q": "4 × 6 = ?",  "a": "24", "options": ["20", "24", "28"], "answer_idx": 1},
    {"q": "18 ÷ 3 = ?", "a": "6",  "options": ["4",  "6",  "8"],  "answer_idx": 1},
    {"q": "11 + 9 = ?", "a": "20", "options": ["18", "20", "22"], "answer_idx": 1},
    {"q": "5 × 7 = ?",  "a": "35", "options": ["30", "35", "40"], "answer_idx": 1},
]


def _update_state(**kwargs):
    with fit_test_lock:
        fit_test_state.update(kwargs)


def _set_phase(phase_name: str, detail: str = ""):
    _update_state(phase=phase_name, phase_detail=detail)


def _record_phase(name: str, passed: bool, score: float, detail: str):
    with fit_test_lock:
        fit_test_state["phases"][name] = {
            "pass":   passed,
            "score":  round(score, 3),
            "detail": detail,
        }


# ── Gaze direction helper ───────────────────────────────────────────────────
def _estimate_gaze(landmarks, frame_w):
    """
    Returns 'left', 'center', 'right' based on iris centre vs eye corners.
    Uses MediaPipe 478-pt landmarks:
      Left iris centre  ≈ 468
      Right iris centre ≈ 473
      Left eye corners  : outer=33, inner=133
      Right eye corners : outer=263, inner=362
    """
    if not landmarks or len(landmarks) < 478:
        return "unknown"

    l_iris_x  = landmarks[468][0]
    r_iris_x  = landmarks[473][0]
    l_out_x   = landmarks[33][0]
    l_in_x    = landmarks[133][0]
    r_in_x    = landmarks[362][0]
    r_out_x   = landmarks[263][0]

    # Normalise iris position within each eye [0=outer, 1=inner]
    l_norm = (l_iris_x - l_out_x) / max(l_in_x - l_out_x, 1)
    r_norm = (r_iris_x - r_in_x)  / max(r_out_x - r_in_x, 1)
    avg    = (l_norm + r_norm) / 2.0

    if avg < 0.35:
        return "left"
    if avg > 0.65:
        return "right"
    return "center"


# ── Main test runner ────────────────────────────────────────────────────────
def run_fit_test(
    shared_state,
    state_lock,
    button_pressed_event,   # threading.Event or None
    buzzer_fn,              # callable(duration_ms)
    led_fn,                 # callable(bool) — fit-test LED on/off
    on_complete_cb,         # callable(result, state_dict)
    socketio=None,          # for emitting fit_test_update events
):
    """
    Blocking function — run in a daemon thread.
    Reads current driver state from shared_state dict (protected by state_lock).
    """

    def snap():
        with state_lock:
            return dict(shared_state)

    def emit_update():
        if socketio:
            with fit_test_lock:
                payload = dict(fit_test_state)
            socketio.emit("fit_test_update", payload)

    def wait_button(timeout_s) -> bool:
        """Wait for button press. Returns True if pressed within timeout."""
        if button_pressed_event is None:
            # No GPIO — simulate auto-press after random delay (for dev/testing)
            delay = random.uniform(0.3, timeout_s * 0.7)
            time.sleep(delay)
            return True
        button_pressed_event.clear()
        return button_pressed_event.wait(timeout=timeout_s)

    # ── Init ─────────────────────────────────────────────────────────────
    _update_state(
        status="running",
        phase=None,
        phase_detail="",
        score=0.0,
        phases={},
        fail_reason="",
        last_run_ts=time.time(),
    )
    emit_update()
    buzzer_beep_short = lambda: buzzer_fn(150)
    phase_scores = {}

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1 — FACE PRESENCE
    # ════════════════════════════════════════════════════════════════════
    _set_phase("face_presence", "Look directly at the camera…")
    emit_update()

    deadline    = time.time() + PHASE_TIMEOUT
    stable_from = None
    passed      = False

    while time.time() < deadline:
        s = snap()
        if s.get("driver_present"):
            if stable_from is None:
                stable_from = time.time()
            held = time.time() - stable_from
            _set_phase("face_presence", f"Hold still… {held:.1f}/{FACE_STABLE_SECS}s")
            emit_update()
            if held >= FACE_STABLE_SECS:
                passed = True
                break
        else:
            stable_from = None
            _set_phase("face_presence", "No face detected — centre yourself in camera")
            emit_update()
        time.sleep(0.1)

    score = 1.0 if passed else 0.0
    _record_phase("face_presence", passed, score,
                  "Face stable ✓" if passed else "Face not detected")
    phase_scores["face_presence"] = score
    buzzer_beep_short()
    emit_update()

    if not passed:
        _finish("failed", "Face not detected — please sit in front of camera",
                phase_scores, on_complete_cb)
        return

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2 — EYE OPENNESS (EAR baseline)
    # ════════════════════════════════════════════════════════════════════
    _set_phase("eye_openness", "Keep your eyes open and look forward…")
    emit_update()

    deadline    = time.time() + PHASE_TIMEOUT
    stable_from = None
    passed      = False
    ear_samples = []

    while time.time() < deadline:
        s = snap()
        ear = s.get("ear", 0.0)
        if s.get("driver_present") and ear >= EAR_OPEN_THRESHOLD:
            ear_samples.append(ear)
            if stable_from is None:
                stable_from = time.time()
            held = time.time() - stable_from
            _set_phase("eye_openness",
                       f"EAR {ear:.3f} — Hold open… {held:.1f}/{EAR_STABLE_SECS}s")
            emit_update()
            if held >= EAR_STABLE_SECS:
                passed = True
                break
        else:
            stable_from = None
            ear_samples = []
            detail = (f"EAR {ear:.3f} — eyes too closed (min {EAR_OPEN_THRESHOLD})"
                      if s.get("driver_present") else "Face lost — look at camera")
            _set_phase("eye_openness", detail)
            emit_update()
        time.sleep(0.1)

    avg_ear = sum(ear_samples)/len(ear_samples) if ear_samples else 0
    score   = min((avg_ear / 0.35) * 1.0, 1.0) if passed else 0.3
    _record_phase("eye_openness", passed, score,
                  f"Avg EAR {avg_ear:.3f} ✓" if passed else f"EAR too low ({avg_ear:.3f})")
    phase_scores["eye_openness"] = score
    buzzer_beep_short()
    emit_update()

    # Non-fatal — continue even if low (contributes to composite score)

    # ════════════════════════════════════════════════════════════════════
    # PHASE 3 — HEAD POSITION
    # ════════════════════════════════════════════════════════════════════
    _set_phase("head_position", "Look straight ahead at the camera…")
    emit_update()

    deadline    = time.time() + PHASE_TIMEOUT
    stable_from = None
    passed      = False

    while time.time() < deadline:
        s    = snap()
        lm   = s.get("landmarks", [])
        fw   = s.get("stream_w", 640)
        fh   = s.get("stream_h", 480)

        if s.get("driver_present") and len(lm) >= 10:
            # Landmark 1 = nose tip
            nx, ny = lm[1][0], lm[1][1]
            cx_norm = abs(nx / fw - 0.5)   # 0 = perfectly centred
            cy_norm = abs(ny / fh - 0.5)
            centred = cx_norm < HEAD_CENTRE_TOL and cy_norm < HEAD_PITCH_TOL

            if centred:
                if stable_from is None:
                    stable_from = time.time()
                held = time.time() - stable_from
                _set_phase("head_position",
                           f"Good position — hold… {held:.1f}/{HEAD_STABLE_SECS}s")
                emit_update()
                if held >= HEAD_STABLE_SECS:
                    passed = True
                    break
            else:
                stable_from = None
                hint = "Turn left" if nx/fw > 0.5 + HEAD_CENTRE_TOL else \
                       "Turn right" if nx/fw < 0.5 - HEAD_CENTRE_TOL else \
                       "Tilt up" if ny/fh > 0.5 + HEAD_PITCH_TOL else "Tilt down"
                _set_phase("head_position", f"{hint} — centre your head")
                emit_update()
        else:
            stable_from = None
            _set_phase("head_position", "Face not detected")
            emit_update()
        time.sleep(0.1)

    score = 1.0 if passed else 0.4
    _record_phase("head_position", passed, score,
                  "Head centred ✓" if passed else "Head not centred")
    phase_scores["head_position"] = score
    buzzer_beep_short()
    emit_update()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 4 — GAZE TRACKING
    # ════════════════════════════════════════════════════════════════════
    gaze_sequence = ["left", "center", "right"]
    gaze_passed   = 0
    gaze_timeout  = time.time() + GAZE_TIMEOUT_SECS

    for direction in gaze_sequence:
        arrow = "← Look LEFT" if direction == "left" else \
                "↑ Look CENTRE" if direction == "center" else "→ Look RIGHT"
        _set_phase("gaze_tracking", arrow)
        emit_update()

        held_from  = None
        dir_passed = False

        while time.time() < gaze_timeout:
            s  = snap()
            lm = s.get("landmarks", [])
            fw = s.get("stream_w", 640)

            if s.get("driver_present") and lm:
                gaze = _estimate_gaze(lm, fw)
                if gaze == direction:
                    if held_from is None:
                        held_from = time.time()
                    held = time.time() - held_from
                    _set_phase("gaze_tracking",
                               f"{arrow} — hold… {held:.1f}/{GAZE_HOLD_SECS}s")
                    emit_update()
                    if held >= GAZE_HOLD_SECS:
                        dir_passed = True
                        gaze_passed += 1
                        buzzer_beep_short()
                        break
                else:
                    held_from = None
                    _set_phase("gaze_tracking", f"{arrow}  (detected: {gaze})")
                    emit_update()
            time.sleep(0.08)

        if not dir_passed:
            break   # timed out

    gaze_score = gaze_passed / len(gaze_sequence)
    passed     = gaze_passed == len(gaze_sequence)
    _record_phase("gaze_tracking", passed, gaze_score,
                  f"{gaze_passed}/{len(gaze_sequence)} directions ✓"
                  if passed else f"Only {gaze_passed}/{len(gaze_sequence)} directions")
    phase_scores["gaze_tracking"] = gaze_score
    emit_update()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 5 — REACTION TIME
    # ════════════════════════════════════════════════════════════════════
    _set_phase("reaction_time", "Get ready — press button when LED flashes…")
    emit_update()
    time.sleep(1.5)

    rt_times = []
    rt_passed = 0

    for rnd in range(REACTION_ROUNDS):
        delay = random.uniform(1.5, 4.0)
        _set_phase("reaction_time",
                   f"Round {rnd+1}/{REACTION_ROUNDS} — wait for flash…")
        emit_update()
        time.sleep(delay)

        # Flash LED
        led_fn(True)
        buzzer_fn(80)
        t_flash = time.time()
        _set_phase("reaction_time",
                   f"⚡ PRESS NOW! Round {rnd+1}/{REACTION_ROUNDS}")
        emit_update()

        pressed = wait_button(REACTION_TIMEOUT)
        rt      = time.time() - t_flash
        led_fn(False)

        if pressed and rt <= REACTION_TIMEOUT:
            rt_times.append(rt)
            rt_passed += 1
            _set_phase("reaction_time",
                       f"✓ {rt*1000:.0f}ms — Round {rnd+1} passed")
        else:
            rt_times.append(REACTION_TIMEOUT)
            _set_phase("reaction_time",
                       f"✗ Too slow — Round {rnd+1}")
        emit_update()
        time.sleep(0.8)

    avg_rt = sum(rt_times) / len(rt_times) if rt_times else REACTION_TIMEOUT
    # Score: 1.0 at ≤300ms, 0.0 at ≥1500ms
    rt_score  = max(0.0, 1.0 - (avg_rt - 0.3) / 1.2)
    rt_score  = min(rt_score, 1.0)
    passed    = rt_passed >= 2   # at least 2/3 rounds
    _record_phase("reaction_time", passed, rt_score,
                  f"Avg {avg_rt*1000:.0f}ms ({rt_passed}/{REACTION_ROUNDS} rounds)"
                  if passed else f"Too slow — avg {avg_rt*1000:.0f}ms")
    phase_scores["reaction_time"] = rt_score
    buzzer_beep_short()
    emit_update()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 6 — MATH QUESTION
    # ════════════════════════════════════════════════════════════════════
    q      = random.choice(_MATH_QUESTIONS)
    opts   = q["options"]
    answer = q["answer_idx"]   # always index 1 (middle option B) in our bank

    _update_state(
        phase="math_question",
        phase_detail=f"{q['q']}  A:{opts[0]}  B:{opts[1]}  C:{opts[2]}",
    )
    # Also embed question into fit_test_state so dashboard can render it
    with fit_test_lock:
        fit_test_state["math_question"] = {
            "question": q["q"],
            "options":  opts,
        }
    emit_update()

    # Encode answer via button press timing:
    #   short press  (<0.6s) = A (index 0)
    #   medium press (0.6–1.5s) = B (index 1)
    #   long press   (>1.5s)    = C (index 2)
    t_start   = time.time()
    deadline  = t_start + MATH_TIMEOUT
    math_pass = False
    chosen    = -1

    if button_pressed_event is None:
        # Dev mode — auto answer correctly
        time.sleep(random.uniform(1.0, 3.0))
        chosen    = answer
        math_pass = True
    else:
        while time.time() < deadline:
            button_pressed_event.clear()
            pressed = button_pressed_event.wait(timeout=deadline - time.time())
            if not pressed:
                break   # timed out
            t_down = time.time()
            # Wait for release (button_pressed_event fires on press, so we
            # do a secondary check by waiting for next event or timeout)
            button_pressed_event.clear()
            held = button_pressed_event.wait(timeout=2.0)
            t_up  = time.time()
            hold_s = t_up - t_down

            if hold_s < 0.6:
                chosen = 0   # A
            elif hold_s < 1.5:
                chosen = 1   # B
            else:
                chosen = 2   # C
            break

        math_pass = (chosen == answer)

    score = 1.0 if math_pass else 0.0
    chosen_lbl = ["A", "B", "C"][chosen] if 0 <= chosen <= 2 else "—"
    correct_lbl = ["A", "B", "C"][answer]
    _record_phase(
        "math_question", math_pass, score,
        f"Answered {chosen_lbl} ({'correct ✓' if math_pass else f'wrong — correct: {correct_lbl}'})"
    )
    phase_scores["math_question"] = score
    buzzer_beep_short()
    emit_update()

    # ════════════════════════════════════════════════════════════════════
    # FINAL SCORING
    # ════════════════════════════════════════════════════════════════════
    composite = sum(
        phase_scores.get(p, 0.0) * w
        for p, w in PHASE_WEIGHTS.items()
    )
    composite = round(composite, 3)

    if composite >= PASS_SCORE:
        _finish("passed", "", phase_scores, on_complete_cb, composite)
    else:
        failed_phases = [
            p for p, s in phase_scores.items() if s < 0.5
        ]
        reason = "Low score in: " + ", ".join(failed_phases) if failed_phases else \
                 f"Composite score {composite:.0%} below threshold"
        _finish("failed", reason, phase_scores, on_complete_cb, composite)


# ── Finish helper ───────────────────────────────────────────────────────────
def _finish(result, reason, phase_scores, callback, composite=0.0):
    _update_state(
        status=result,
        score=composite,
        fail_reason=reason,
        last_result=result,
        last_run_ts=time.time(),
        phase=None,
        phase_detail="",
    )
    print(f"[FitTest] {result.upper()} — score={composite:.0%}"
          + (f" — {reason}" if reason else ""))
    try:
        with fit_test_lock:
            state_copy = dict(fit_test_state)
        callback(result, state_copy)
    except Exception as e:
        print(f"[FitTest] Callback error: {e}")
