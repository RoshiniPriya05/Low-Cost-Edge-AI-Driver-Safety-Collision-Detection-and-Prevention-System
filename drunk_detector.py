"""
drunk_detector.py — Impairment Detection
=========================================
Fuses two signal sources:
  1. MQ-3 alcohol sensor via MCP3008 SPI ADC  (60% weight)
  2. Camera heuristics via MediaPipe           (40% weight)
     • EAR instability  (variance over rolling window)
     • Abnormal blink rate
     • Head sway        (nose tip X variance)
     • MAR spikes       (involuntary mouth opening)

Thresholds:
  ≥ 0.40  → drunk_warning  (buzzer only)
  ≥ 0.60  → drunk_detected (buzzer + Bolna call + WhatsApp)

MCP3008 wiring (SPI0):
  MCP3008 VDD  → 3.3V
  MCP3008 VREF → 3.3V
  MCP3008 AGND → GND
  MCP3008 DGND → GND
  MCP3008 CLK  → GPIO 11 (SCLK)
  MCP3008 DOUT → GPIO 9  (MISO)
  MCP3008 DIN  → GPIO 10 (MOSI)
  MCP3008 CS   → GPIO 8  (CE0)
  MQ-3 AOUT   → MCP3008 CH0
"""

import time
import threading
import collections

# ── MCP3008 via spidev ─────────────────────────────────────────────────────
try:
    import spidev
    _spi = spidev.SpiDev()
    _spi.open(0, 0)          # bus=0, device=0 (CE0)
    _spi.max_speed_hz = 1_350_000
    _spi.mode = 0b00
    _sensor_available = True
    print("[Drunk] MCP3008 SPI opened on bus 0, CE0")
except Exception as e:
    _spi = None
    _sensor_available = False
    print(f"[Drunk] MCP3008 not available ({e}) — camera-only mode")


# ── Config ──────────────────────────────────────────────────────────────────
MQ3_CHANNEL          = 0       # MCP3008 channel 0
MQ3_WARN_VOLTAGE     = 0.8     # volts (~clean air baseline ~0.4V)
MQ3_ALERT_VOLTAGE    = 1.6     # volts (significant alcohol presence)
VREF                 = 3.3     # reference voltage

SENSOR_WEIGHT        = 0.60
CAMERA_WEIGHT        = 0.40

WARN_THRESHOLD       = 0.40
ALERT_THRESHOLD      = 0.60

WINDOW               = 60      # frames for rolling camera stats (~3s at 20fps)
EAR_VAR_THRESH       = 0.0012  # EAR variance indicating instability
HEAD_SWAY_THRESH     = 8.0     # pixels std-dev of nose X
MAR_SPIKE_THRESH     = 0.55    # sudden mouth open
BLINK_RATE_LOW       = 5       # blinks/min below normal
BLINK_RATE_HIGH      = 35      # blinks/min above normal


# ── Internal state ──────────────────────────────────────────────────────────
_lock = threading.Lock()

_ear_window    = collections.deque(maxlen=WINDOW)
_nosex_window  = collections.deque(maxlen=WINDOW)
_mar_window    = collections.deque(maxlen=WINDOW)
_blink_times   = collections.deque(maxlen=30)   # timestamps of blinks
_last_eye_open = True
_frame_count   = 0

_drunk_score    = 0.0
_drunk_detected = False
_drunk_warning  = False
_sensor_reading = 0.0          # raw voltage
_sensor_score   = 0.0          # normalised 0-1
_camera_score   = 0.0
_last_read_ts   = 0.0


# ── MCP3008 read ─────────────────────────────────────────────────────────────

def _read_mcp3008(channel: int) -> float:
    """Read raw 10-bit ADC value from MCP3008 channel. Returns voltage."""
    if not _sensor_available or _spi is None:
        return 0.0
    try:
        cmd  = [1, (8 + channel) << 4, 0]
        resp = _spi.xfer2(cmd)
        raw  = ((resp[1] & 3) << 8) | resp[2]
        return (raw / 1023.0) * VREF
    except Exception:
        return 0.0


def _sensor_voltage_to_score(voltage: float) -> float:
    """Map MQ-3 voltage to 0–1 impairment score."""
    if voltage <= MQ3_WARN_VOLTAGE:
        return 0.0
    if voltage >= MQ3_ALERT_VOLTAGE:
        return 1.0
    return (voltage - MQ3_WARN_VOLTAGE) / (MQ3_ALERT_VOLTAGE - MQ3_WARN_VOLTAGE)


# ── Camera heuristics ─────────────────────────────────────────────────────────

def _compute_camera_score() -> float:
    """
    Returns 0–1 impairment score from camera signals.
    Higher = more likely impaired.
    """
    score = 0.0
    signals = 0

    # 1. EAR variance (erratic blinking / drooping)
    if len(_ear_window) >= 20:
        import statistics
        ear_var = statistics.variance(_ear_window)
        if ear_var > EAR_VAR_THRESH:
            score   += min(ear_var / (EAR_VAR_THRESH * 4), 1.0) * 0.30
            signals += 1

    # 2. Head sway (nose X std-dev)
    if len(_nosex_window) >= 20:
        import statistics
        sway = statistics.stdev(_nosex_window)
        if sway > HEAD_SWAY_THRESH:
            score   += min(sway / (HEAD_SWAY_THRESH * 3), 1.0) * 0.30
            signals += 1

    # 3. MAR spikes (involuntary mouth open ≠ yawn)
    if len(_mar_window) >= 10:
        spike_count = sum(1 for m in _mar_window if m > MAR_SPIKE_THRESH)
        if spike_count > 3:
            score   += min(spike_count / 10, 1.0) * 0.20
            signals += 1

    # 4. Abnormal blink rate
    now = time.time()
    recent_blinks = [t for t in _blink_times if now - t <= 60]
    bpm = len(recent_blinks)
    if bpm < BLINK_RATE_LOW or bpm > BLINK_RATE_HIGH:
        score   += 0.20
        signals += 1

    # Normalise — require at least 2 signals to avoid false positives
    if signals < 2:
        score *= 0.4

    return min(score, 1.0)


# ── Public API ────────────────────────────────────────────────────────────────

def update(frame_state: dict):
    """
    Call every AI frame with a dict containing:
      ear, mar, head_nod, eye_status, driver_present,
      nose_x (optional — nose tip pixel X)
    """
    global _drunk_score, _drunk_detected, _drunk_warning
    global _sensor_reading, _sensor_score, _camera_score
    global _last_eye_open, _frame_count, _last_read_ts

    if not frame_state.get("driver_present", False):
        return

    ear       = frame_state.get("ear", 0.3)
    mar       = frame_state.get("mar", 0.0)
    nose_x    = frame_state.get("nose_x", 320)
    eye_open  = frame_state.get("eye_status", "Open") == "Open"

    with _lock:
        _frame_count += 1
        _ear_window.append(ear)
        _mar_window.append(mar)
        _nosex_window.append(nose_x)

        # Blink detection
        if _last_eye_open and not eye_open:
            _blink_times.append(time.time())
        _last_eye_open = eye_open

        # Read sensor every 0.5s (not every frame — SPI is slow)
        now = time.time()
        if now - _last_read_ts >= 0.5:
            voltage         = _read_mcp3008(MQ3_CHANNEL)
            _sensor_reading = round(voltage, 3)
            _sensor_score   = _sensor_voltage_to_score(voltage)
            _last_read_ts   = now

        # Camera score
        _camera_score = _compute_camera_score()

        # Fused score
        fused = (_sensor_score * SENSOR_WEIGHT) + (_camera_score * CAMERA_WEIGHT)
        _drunk_score    = round(min(fused, 1.0), 3)
        _drunk_warning  = _drunk_score >= WARN_THRESHOLD
        _drunk_detected = _drunk_score >= ALERT_THRESHOLD


def get_state() -> dict:
    with _lock:
        return {
            "drunk_score":    _drunk_score,
            "drunk_detected": _drunk_detected,
            "drunk_warning":  _drunk_warning,
            "sensor_active":  _sensor_available,
            "sensor_reading": _sensor_reading,
            "sensor_score":   _sensor_score,
            "camera_score":   _camera_score,
        }


def reset():
    """Call at session start (after fit test passes)."""
    global _drunk_score, _drunk_detected, _drunk_warning
    global _sensor_reading, _sensor_score, _camera_score, _frame_count
    with _lock:
        _ear_window.clear()
        _nosex_window.clear()
        _mar_window.clear()
        _blink_times.clear()
        _drunk_score    = 0.0
        _drunk_detected = False
        _drunk_warning  = False
        _sensor_reading = 0.0
        _sensor_score   = 0.0
        _camera_score   = 0.0
        _frame_count    = 0
    print("[Drunk] State reset for new session")


def cleanup():
    """Call on shutdown."""
    if _sensor_available and _spi:
        try:
            _spi.close()
        except Exception:
            pass
