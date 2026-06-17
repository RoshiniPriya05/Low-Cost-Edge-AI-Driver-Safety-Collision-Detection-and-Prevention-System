"""
notifier.py — Notification Dispatcher
=======================================
Channels:
  • Twilio WhatsApp → supervisor only
      - Fit test result (pass + fail)
      - Critical fatigue (5/5 events)
      - Drunk detected (score ≥ 60%)

  • Bolna AI Voice Call → driver + supervisor
      - Fit test FAIL only
      - Critical fatigue
      - Drunk detected

Dedup:
  Per-session keyed by (trigger, role).
  reset_session() clears on new fit-test pass.

Dry-run:
  Set NOTIFIER_DRY_RUN=true in env to log without sending.
"""

import os
import time
import threading
import logging
from datetime import datetime

logger = logging.getLogger("notifier")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)

# ── Config (from environment) ──────────────────────────────────────────────
CFG = {
    # Twilio
    "TWILIO_ACCOUNT_SID":   os.getenv("TWILIO_ACCOUNT_SID",   ""),
    "TWILIO_AUTH_TOKEN":    os.getenv("TWILIO_AUTH_TOKEN",     ""),
    "TWILIO_WA_FROM":       os.getenv("TWILIO_WHATSAPP_FROM",  "whatsapp:+14155238886"),

    # Recipients
    "DRIVER_PHONE":         os.getenv("DRIVER_PHONE",          ""),   # whatsapp:+91XXXXXXXXXX
    "SUPERVISOR_PHONE":     os.getenv("SUPERVISOR_PHONE",      ""),   # whatsapp:+91XXXXXXXXXX

    # Bolna
    "BOLNA_API_KEY":            os.getenv("BOLNA_API_KEY",             ""),
    "BOLNA_AGENT_ID_DRIVER":    os.getenv("BOLNA_AGENT_ID_DRIVER",     ""),
    "BOLNA_AGENT_ID_SUPERVISOR":os.getenv("BOLNA_AGENT_ID_SUPERVISOR", ""),
    "BOLNA_BASE_URL":           os.getenv("BOLNA_BASE_URL",            "https://api.bolna.dev/call"),

    # Mode
    "DRY_RUN": os.getenv("NOTIFIER_DRY_RUN", "false").lower() == "true",
}

# ── Notification log ───────────────────────────────────────────────────────
_log: list  = []
_log_lock   = threading.Lock()
_MAX_LOG    = 150

# ── Session dedup ──────────────────────────────────────────────────────────
_fired:      set = set()
_fired_lock       = threading.Lock()


def reset_session():
    with _fired_lock:
        _fired.clear()
    logger.info("Session reset — dedup cleared")


def _already_fired(trigger: str, role: str) -> bool:
    key = f"{trigger}:{role}"
    with _fired_lock:
        if key in _fired:
            return True
        _fired.add(key)
        return False


# ── WhatsApp message templates ─────────────────────────────────────────────
_WA_TEMPLATES = {
    "fit_test_pass": (
        "✅ *FLEET SAFETY — FIT TEST PASSED*\n"
        "Driver: {driver_id}\n"
        "Score: {score_pct}%\n"
        "Time: {time}\n"
        "Session started successfully."
    ),
    "fit_test_fail": (
        "🚫 *FLEET SAFETY — FIT TEST FAILED*\n"
        "Driver: {driver_id}\n"
        "Reason: {reason}\n"
        "Score: {score_pct}%\n"
        "Time: {time}\n"
        "Vehicle remains locked. Driver must retry or request override."
    ),
    "critical_fatigue": (
        "⚠️ *FLEET SAFETY — CRITICAL FATIGUE*\n"
        "Driver: {driver_id}\n"
        "Fatigue events: {events}/5\n"
        "Session uptime: {uptime}\n"
        "Time: {time}\n"
        "Driver has been alerted via AI call. Monitor situation."
    ),
    "drunk_detected": (
        "🚨 *FLEET SAFETY — POSSIBLE IMPAIRMENT*\n"
        "Driver: {driver_id}\n"
        "Impairment score: {score_pct}%\n"
        "Session uptime: {uptime}\n"
        "Time: {time}\n"
        "Driver has been alerted via AI call. Advise immediate stop."
    ),
}


# ── Internal log helper ────────────────────────────────────────────────────
def _add_log(trigger, channel, role, recipient, status, detail=""):
    entry = {
        "ts":        time.time(),
        "time":      datetime.now().strftime("%H:%M:%S"),
        "trigger":   trigger,
        "channel":   channel,
        "role":      role,
        "recipient": recipient or "—",
        "status":    status,
        "detail":    detail,
    }
    with _log_lock:
        _log.append(entry)
        if len(_log) > _MAX_LOG:
            _log.pop(0)
    logger.info(
        f"[{channel:<12}] {trigger:<22} {role:<10} → "
        f"{(recipient or 'unset'):<20} : {status}"
        + (f"  ({detail[:60]})" if detail else "")
    )


def get_log() -> list:
    with _log_lock:
        return list(_log)


# ── Twilio WhatsApp ────────────────────────────────────────────────────────
def _send_whatsapp(to_phone: str, body: str, trigger: str, role: str) -> bool:
    if not to_phone:
        _add_log(trigger, "whatsapp", role, to_phone, "skipped", "no phone configured")
        return False

    if CFG["DRY_RUN"]:
        _add_log(trigger, "whatsapp", role, to_phone, "dry_run")
        logger.info(f"[DRY RUN] WhatsApp to {to_phone}:\n{body}")
        return True

    try:
        from twilio.rest import Client
        client  = Client(CFG["TWILIO_ACCOUNT_SID"], CFG["TWILIO_AUTH_TOKEN"])
        message = client.messages.create(
            from_=CFG["TWILIO_WA_FROM"],
            to=to_phone,
            body=body,
        )
        _add_log(trigger, "whatsapp", role, to_phone, "sent", message.sid)
        return True
    except Exception as exc:
        _add_log(trigger, "whatsapp", role, to_phone, "failed", str(exc))
        return False


# ── Bolna AI call ──────────────────────────────────────────────────────────
def _make_bolna_call(
    to_phone:  str,
    agent_id:  str,
    user_data: dict,
    trigger:   str,
    role:      str,
) -> bool:
    if not to_phone:
        _add_log(trigger, "bolna_call", role, to_phone, "skipped", "no phone configured")
        return False

    if not CFG["BOLNA_API_KEY"] or not agent_id:
        _add_log(trigger, "bolna_call", role, to_phone, "skipped", "no API key / agent ID")
        return False

    if CFG["DRY_RUN"]:
        _add_log(trigger, "bolna_call", role, to_phone, "dry_run")
        logger.info(f"[DRY RUN] Bolna call to {to_phone} agent={agent_id} data={user_data}")
        return True

    try:
        import requests
        payload = {
            "agent_id":                agent_id,
            "recipient_phone_number":  to_phone,
            "user_data":               user_data,
        }
        headers = {
            "Authorization": f"Bearer {CFG['BOLNA_API_KEY']}",
            "Content-Type":  "application/json",
        }
        resp = requests.post(
            CFG["BOLNA_BASE_URL"],
            json=payload,
            headers=headers,
            timeout=15,
        )
        if resp.ok:
            _add_log(trigger, "bolna_call", role, to_phone, "sent",
                     resp.json().get("call_id", ""))
            return True
        _add_log(trigger, "bolna_call", role, to_phone, "failed",
                 f"HTTP {resp.status_code}: {resp.text[:80]}")
        return False
    except Exception as exc:
        _add_log(trigger, "bolna_call", role, to_phone, "failed", str(exc))
        return False


# ── Bolna user_data builder ────────────────────────────────────────────────
def _bolna_vars(trigger: str, role: str, ctx: dict) -> dict:
    """Variables injected into Bolna agent script."""
    return {
        "trigger":    trigger,
        "role":       role,
        "driver_id":  ctx.get("driver_id", "Unknown"),
        "time":       ctx.get("time",       "—"),
        "uptime":     ctx.get("uptime",     "N/A"),
        "reason":     ctx.get("reason",     "N/A"),
        "score_pct":  str(ctx.get("score_pct", "N/A")),
        "events":     str(ctx.get("events",    "N/A")),
    }


# ── Dispatch helpers ───────────────────────────────────────────────────────
def _dispatch_supervisor_wa(trigger: str, ctx: dict):
    if _already_fired(trigger, "supervisor_wa"):
        return
    body = _WA_TEMPLATES.get(trigger, "Fleet Safety Alert — {driver_id}").format(**ctx)
    _send_whatsapp(CFG["SUPERVISOR_PHONE"], body, trigger, "supervisor")


def _dispatch_driver_call(trigger: str, ctx: dict):
    if _already_fired(trigger, "driver_call"):
        return
    # Driver gets a Bolna AI call — no WhatsApp (driving, hands on wheel)
    _make_bolna_call(
        _driver_raw_phone(),
        CFG["BOLNA_AGENT_ID_DRIVER"],
        _bolna_vars(trigger, "driver", ctx),
        trigger,
        "driver",
    )


def _dispatch_supervisor_call(trigger: str, ctx: dict):
    if _already_fired(trigger, "supervisor_call"):
        return
    _make_bolna_call(
        _supervisor_raw_phone(),
        CFG["BOLNA_AGENT_ID_SUPERVISOR"],
        _bolna_vars(trigger, "supervisor", ctx),
        trigger,
        "supervisor",
    )


def _driver_raw_phone() -> str:
    """Strip 'whatsapp:' prefix for Bolna (needs plain E.164)."""
    p = CFG["DRIVER_PHONE"]
    return p.replace("whatsapp:", "") if p else ""


def _supervisor_raw_phone() -> str:
    p = CFG["SUPERVISOR_PHONE"]
    return p.replace("whatsapp:", "") if p else ""


def _fire_background(fns):
    """Run a list of callables in background threads."""
    for fn in fns:
        threading.Thread(target=fn, daemon=True).start()


# ── Public notification functions ──────────────────────────────────────────

def notify_fit_test_pass(driver_id: str, score: float, uptime_s: float = 0):
    """SMS to supervisor only on pass. No calls needed."""
    ctx = {
        "driver_id": driver_id,
        "time":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score_pct": int(score * 100),
        "uptime":    _fmt_uptime(uptime_s),
        "reason":    "All phases passed",
        "events":    "N/A",
    }
    _fire_background([
        lambda: _dispatch_supervisor_wa("fit_test_pass", ctx),
    ])


def notify_fit_test_fail(driver_id: str, reason: str, score: float, uptime_s: float = 0):
    """SMS to supervisor + Bolna call to driver AND supervisor on fail."""
    ctx = {
        "driver_id": driver_id,
        "time":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score_pct": int(score * 100),
        "uptime":    _fmt_uptime(uptime_s),
        "reason":    reason,
        "events":    "N/A",
    }
    _fire_background([
        lambda: _dispatch_supervisor_wa("fit_test_fail", ctx),
        lambda: _dispatch_driver_call("fit_test_fail", ctx),
        lambda: _dispatch_supervisor_call("fit_test_fail", ctx),
    ])


def notify_critical_fatigue(driver_id: str, events: int, uptime_s: float):
    """SMS to supervisor + Bolna call to driver AND supervisor."""
    ctx = {
        "driver_id": driver_id,
        "time":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "events":    events,
        "uptime":    _fmt_uptime(uptime_s),
        "reason":    f"{events} fatigue events detected",
        "score_pct": "N/A",
    }
    _fire_background([
        lambda: _dispatch_supervisor_wa("critical_fatigue", ctx),
        lambda: _dispatch_driver_call("critical_fatigue", ctx),
        lambda: _dispatch_supervisor_call("critical_fatigue", ctx),
    ])


def notify_drunk_detected(driver_id: str, score: float, uptime_s: float):
    """SMS to supervisor + Bolna call to driver AND supervisor."""
    ctx = {
        "driver_id": driver_id,
        "time":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score_pct": int(score * 100),
        "uptime":    _fmt_uptime(uptime_s),
        "reason":    f"Impairment score {int(score*100)}%",
        "events":    "N/A",
    }
    _fire_background([
        lambda: _dispatch_supervisor_wa("drunk_detected", ctx),
        lambda: _dispatch_driver_call("drunk_detected", ctx),
        lambda: _dispatch_supervisor_call("drunk_detected", ctx),
    ])


# ── Util ───────────────────────────────────────────────────────────────────
def _fmt_uptime(s: float) -> str:
    s   = int(s)
    h   = s // 3600
    m   = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"
