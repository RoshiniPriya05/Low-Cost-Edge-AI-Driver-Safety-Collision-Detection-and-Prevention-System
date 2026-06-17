"""
ignition_relay.py — Software-only Ignition Relay Manager
=========================================================
No physical relay hardware. This module manages a software lock state
that freezes the dashboard and triggers the buzzer.

When hardware relay is added later:
  1. Uncomment the GPIO section at the bottom
  2. Set RELAY_GPIO_PIN to your relay's BCM pin
  3. The rest of the logic stays identical

States:
  ARMED      → system ready, driver can drive
  WARNING    → first alert fired, buzzer beeping
  COUNTDOWN  → escalation in progress (10s before soft-lock)
  LOCKED     → system locked, driver must retry or supervisor unlocks
"""

import threading
import time

# ── Hardware relay stub ────────────────────────────────────────────────────
# Uncomment this block when physical relay is wired up:
#
# RELAY_GPIO_PIN = 26   # BCM pin connected to relay IN
# try:
#     import lgpio
#     from gpiozero import OutputDevice
#     from gpiozero.pins.lgpio import LGPIOFactory
#     from gpiozero import Device
#     Device.pin_factory = LGPIOFactory(chip=0)
#     _relay_device = OutputDevice(RELAY_GPIO_PIN, active_high=False, initial_value=False)
#     _hw_relay_available = True
# except Exception as e:
#     print(f"[Relay] Hardware relay init failed: {e}")
#     _relay_device = None
#     _hw_relay_available = False

_hw_relay_available = False
_relay_device = None


# ── Internal state ─────────────────────────────────────────────────────────
_state = {
    "locked":            True,   # starts locked until fit test passes
    "lock_reason":       "System startup — awaiting fit test",
    "escalation_phase":  "idle", # idle | warning | countdown | locked
    "countdown_s":       0,
    "locked_at":         None,
}
_state_lock      = threading.Lock()
_countdown_timer = None
_socketio_ref    = None   # injected by app.py


def setup(socketio=None):
    """Call once at startup. Pass socketio instance for countdown events."""
    global _socketio_ref
    _socketio_ref = socketio
    _set_hw_relay(True)   # start locked
    print("[Relay] Software relay initialised — LOCKED (awaiting fit test)")


# ── Public API ─────────────────────────────────────────────────────────────

def lock_immediate(reason: str = "unspecified"):
    """Lock immediately — no countdown."""
    global _countdown_timer
    _cancel_countdown()
    with _state_lock:
        _state["locked"]           = True
        _state["lock_reason"]      = reason
        _state["escalation_phase"] = "locked"
        _state["countdown_s"]      = 0
        _state["locked_at"]        = time.time()
    _set_hw_relay(True)
    print(f"[Relay] LOCKED — {reason}")


def unlock(reason: str = "unspecified"):
    """Unlock — arm the system."""
    _cancel_countdown()
    with _state_lock:
        _state["locked"]           = False
        _state["lock_reason"]      = ""
        _state["escalation_phase"] = "idle"
        _state["countdown_s"]      = 0
        _state["locked_at"]        = None
    _set_hw_relay(False)
    print(f"[Relay] ARMED — {reason}")


def start_escalation(warning_cb=None, lock_cb=None, countdown_s: int = 10):
    """
    Begin escalation sequence:
      phase 1 — WARNING  (fires warning_cb immediately)
      phase 2 — COUNTDOWN for countdown_s seconds
      phase 3 — LOCKED   (fires lock_cb)
    The driver can cancel by calling cancel_escalation() within the window.
    """
    global _countdown_timer
    _cancel_countdown()

    with _state_lock:
        _state["escalation_phase"] = "warning"
        _state["countdown_s"]      = countdown_s

    if warning_cb:
        try:
            warning_cb()
        except Exception:
            pass

    print(f"[Relay] Escalation started — locking in {countdown_s}s")
    _countdown_timer = threading.Thread(
        target=_run_countdown,
        args=(countdown_s, lock_cb),
        daemon=True,
    )
    _countdown_timer.start()


def cancel_escalation():
    """Driver pressed button — cancel pending lock."""
    _cancel_countdown()
    with _state_lock:
        if _state["escalation_phase"] in ("warning", "countdown"):
            _state["escalation_phase"] = "idle"
            _state["countdown_s"]      = 0
    print("[Relay] Escalation cancelled by driver")


def get_state() -> dict:
    with _state_lock:
        return dict(_state)


def is_locked() -> bool:
    with _state_lock:
        return _state["locked"]


# ── Internal ───────────────────────────────────────────────────────────────

def _run_countdown(seconds: int, lock_cb):
    for remaining in range(seconds, 0, -1):
        with _state_lock:
            if _state["escalation_phase"] not in ("warning", "countdown"):
                return   # cancelled
            _state["escalation_phase"] = "countdown"
            _state["countdown_s"]      = remaining

        if _socketio_ref:
            try:
                _socketio_ref.emit("relay_countdown", {"seconds": remaining})
            except Exception:
                pass

        time.sleep(1)

    # Check again — may have been cancelled during last sleep
    with _state_lock:
        if _state["escalation_phase"] != "countdown":
            return

    lock_immediate("Auto-lock after countdown")
    if lock_cb:
        try:
            lock_cb()
        except Exception:
            pass


def _cancel_countdown():
    global _countdown_timer
    if _countdown_timer and _countdown_timer.is_alive():
        # Signal cancellation via state — thread checks each second
        with _state_lock:
            if _state["escalation_phase"] in ("warning", "countdown"):
                _state["escalation_phase"] = "cancelled"
        _countdown_timer = None


def _set_hw_relay(locked: bool):
    """Activate/deactivate physical relay if wired."""
    if not _hw_relay_available or _relay_device is None:
        return
    try:
        if locked:
            _relay_device.on()    # relay energised = ignition cut
        else:
            _relay_device.off()   # relay de-energised = ignition armed
    except Exception as e:
        print(f"[Relay] HW relay error: {e}")
