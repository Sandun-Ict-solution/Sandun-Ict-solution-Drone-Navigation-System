# ══════════════════════════════════════════════════════════════════════════════
#  TELLO AUTO DRIVE — Autonomous Path-Following Inspection Drone
#  Combined from app.py + app_fixed.py, stripped to a deployable field build.
#
#  Tabs:  CONTROL · LIVE CAMERA · AUTO DRIVE · TELEMETRY
#  Removed: AI AUTOPILOT(renamed)/ANA AI / ANALYTICS / REPORT / DEFECTS /
#           CONFIG / GALLERY / WEATHER  (per requested scope)
#
#  AUTO DRIVE lets you draw an exact waypoint path on a grid. The mission
#  thread flies that path — and ONLY that path: if the front sensor reports
#  a wall/obstacle closer than the safe distance, the drone HOLDS position
#  and retries; it never reroutes onto a different path. If still blocked
#  after the retry window, the mission aborts and lands in place rather
#  than improvising a new route.
#
#  Hardware note: the retail DJI Tello only reports a DOWNWARD time-of-flight
#  (ToF) reading. Real forward/side wall-distance sensing (as assumed for
#  this build) requires a Tello EDU + Mission Pads, or an external
#  rangefinder feeding `push_alert`/`tel["tof"]`. The code treats
#  `tel["tof"]`/`get_front_distance_cm()` as a generic "distance to
#  obstacle in the direction of travel" reading — swap `get_front_distance_cm()`
#  with your real sensor call if you're on different hardware.
#
#  Requirements:
#      pip install streamlit djitellopy opencv-python numpy pandas
#      (optional) pip install plotly streamlit-autorefresh
#
#  Run:
#      streamlit run app.py
# ══════════════════════════════════════════════════════════════════════════════

# ── Standard library ──────────────────────────────────────────────────────────
import streamlit as st
import threading
import queue
import time
import datetime
import math
import json
import csv
import io
import base64
import uuid
import os

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── Hardware ──────────────────────────────────────────────────────────────────
try:
    from djitellopy import Tello
    TELLO_AVAILABLE = True
except ImportError:
    TELLO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import streamlit.components.v1 as _stc

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Tello Auto Drive",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL QUEUES & MJPEG FRAME BUFFER
# ══════════════════════════════════════════════════════════════════════════════
ALERT_Q: queue.Queue = queue.Queue(maxsize=50)

_MJPEG_LOCK      = threading.Lock()
_MJPEG_FRAME: bytes = b""
_MJPEG_META: dict   = {}
_MJPEG_PORT      = 8889
_MJPEG_STARTED   = False

VIDEO_DIR = "recordings"
os.makedirs(VIDEO_DIR, exist_ok=True)


def get_live_frame_b64() -> str:
    """Return the latest camera frame as a base64 JPEG string."""
    with _MJPEG_LOCK:
        frame = bytes(_MJPEG_FRAME)
    if not frame:
        return ""
    return base64.b64encode(frame).decode()


def live_camera_component(height: int = 440):
    """True live camera feed: injects the latest frame instantly, then polls
    the local MJPEG snapshot endpoint in JS for near-real-time updates."""
    b64 = get_live_frame_b64()
    initial_src = f"data:image/jpeg;base64,{b64}" if b64 else ""

    html = f"""
<!DOCTYPE html><html><body style="margin:0;padding:0;background:#060B12;overflow:hidden">
<div style="position:relative;width:100%;height:{height}px">
  <div style="position:absolute;top:6px;left:8px;z-index:10;
       font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#C0392B;
       background:rgba(0,0,0,0.8);padding:2px 8px;border-radius:3px;
       border:1px solid #C0392B;letter-spacing:2px;
       animation:br 1s ease-in-out infinite">&#9679; LIVE</div>
  <img id="lf" src="{initial_src}"
       style="width:100%;height:{height}px;display:block;object-fit:contain;
              background:#060B12"
       alt="Starting stream…">
  <div id="st" style="position:absolute;bottom:6px;right:8px;
       font-family:monospace;font-size:0.58rem;color:#2E86AB;
       background:rgba(0,0,0,0.65);padding:2px 6px;border-radius:3px"></div>
</div>
<style>
  @keyframes br{{0%,100%{{opacity:1}}50%{{opacity:0.25}}}}
  body{{background:#060B12}}
</style>
<script>
(function(){{
  var img   = document.getElementById('lf');
  var stats = document.getElementById('st');
  var port  = {_MJPEG_PORT};
  var t0    = Date.now(), frames=0, errs=0, useFetch=true;

  function setFrame(src) {{
    img.src = src;
    frames++;
    var fps = (frames / ((Date.now()-t0)/1000)).toFixed(1);
    stats.textContent = fps + ' fps';
  }}

  function fetchFrame() {{
    if (!useFetch) return;
    fetch('http://localhost:' + port + '/snapshot?_=' + Date.now(), {{cache:'no-store'}})
      .then(function(r) {{ if (!r.ok) throw new Error(r.status); return r.blob(); }})
      .then(function(blob) {{
        var reader = new FileReader();
        reader.onloadend = function() {{ setFrame(reader.result); }};
        reader.readAsDataURL(blob);
        errs = 0;
      }})
      .catch(function(e) {{
        errs++;
        if (errs >= 5) {{ useFetch = false; stats.textContent = 'static (refresh page)'; }}
      }});
  }}

  fetchFrame();
  var iv = setInterval(function() {{
    if (useFetch) fetchFrame(); else clearInterval(iv);
  }}, 80);
}})();
</script>
</body></html>
"""
    _stc.html(html, height=height + 4, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CAM_FILTERS  = ["Normal", "Grayscale", "Edge Detection", "Night Vision", "Thermal"]
PATH_PRESETS = ["Draw Path (manual)", "Grid Scan", "Perimeter Loop", "Zigzag"]

# ══════════════════════════════════════════════════════════════════════════════
#  CSS — Futuristic HUD theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
body { background: #0B1420; color: #D5DEE6; }
.stApp { background: #0B1420; }
section[data-testid="stSidebar"] { background: #101B26; }

@keyframes headerSweep { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
@keyframes fadeInUp { 0%{opacity:0; transform:translateY(6px)} 100%{opacity:1; transform:translateY(0)} }
@keyframes breathe { 0%,100%{opacity:1} 50%{opacity:0.55} }

.hud-header {
  background: linear-gradient(120deg, #0A1118 0%, #12222F 35%, #16303F 55%, #0A1118 100%);
  background-size: 200% 100%;
  animation: fadeInUp 0.5s ease-out, gradientDrift 12s ease-in-out infinite;
  border-bottom: 1px solid #1C3D52; border-top: 1px solid #1C3D52;
  padding: 10px 24px; margin-bottom: 10px; position: relative; overflow: hidden;
}
@keyframes gradientDrift { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
.hud-header::before { content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg, transparent, rgba(46,134,171,0.9), transparent);
  background-size: 50% 100%; animation: headerSweep 4s linear infinite; }
.hud-title { font-family:'Inter',sans-serif; font-size:1.8rem; font-weight:800;
  letter-spacing:1.5px; color:#2E86AB; text-shadow:0 0 16px rgba(46,134,171,0.35); }
.hud-subtitle { font-size:0.72rem; color:#6C89A0; letter-spacing:2px;
  text-transform:uppercase; margin-top:2px; }
.hud-version { font-family:'IBM Plex Mono',monospace; color:#27AE60;
  font-size:0.75rem; letter-spacing:0.5px; }

.s-pill { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; letter-spacing:1px;
  padding:4px 12px; border-radius:3px; font-weight:600; border:1px solid; text-transform:uppercase; }
.s-on   { color:#27AE60; border-color:#27AE60; background:rgba(39,174,96,0.08); }
.s-off  { color:#43606D; border-color:#24404F; background:rgba(0,0,0,0.3); }
.s-warn { color:#E67E22; border-color:#E67E22; background:rgba(230,126,34,0.08); }
.s-crit { color:#C0392B; border-color:#C0392B; background:rgba(192,57,43,0.1);
          animation:pulse-red 1.5s ease-in-out infinite; }
@keyframes pulse-red { 0%,100%{opacity:1} 50%{opacity:0.6} }

.kpi-card { background:linear-gradient(135deg,#101B26 0%,#152736 100%);
  border:1px solid #1C3D52; border-radius:6px; padding:12px 14px; text-align:center;
  box-shadow:0 0 12px rgba(46,134,171,0.05); position:relative; overflow:hidden;
  animation: fadeInUp 0.4s ease-out; transition: transform 0.2s ease, box-shadow 0.2s ease; }
.kpi-card:hover { transform: translateY(-2px); box-shadow:0 4px 16px rgba(46,134,171,0.12); }
.kpi-card::after { content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,transparent,#2E86AB,transparent); }
.kpi-val { font-family:'IBM Plex Mono',monospace; font-size:1.5rem; font-weight:700; }
.kpi-lbl { font-size:0.6rem; color:#6C89A0; text-transform:uppercase; letter-spacing:2px; margin-top:2px; }
.kpi-sub { font-size:0.65rem; color:#4A6F7D; margin-top:2px; }

.sec-hdr { font-family:'Inter',sans-serif; font-size:0.85rem; font-weight:600;
  letter-spacing:2px; color:#2E86AB; text-transform:uppercase;
  border-bottom:1px solid #1C3D52; padding-bottom:5px; margin:12px 0 8px;
  display:flex; align-items:center; gap:6px; }

.cam-panel { background:#060B12; border:1px solid #1C3D52; border-radius:8px;
  overflow:hidden; box-shadow:0 0 30px rgba(46,134,171,0.08); position:relative; }
.cam-offline { display:flex; align-items:center; justify-content:center; height:180px;
  flex-direction:column; gap:8px; color:#24505F; font-family:'IBM Plex Mono',monospace;
  font-size:0.75rem; letter-spacing:1px;
  background:repeating-linear-gradient(45deg,#060B12,#060B12 10px,#050c14 10px,#050c14 20px); }

.stTabs [data-baseweb="tab-list"] { gap:4px; background:#101B26; border-radius:6px;
  padding:4px; border:1px solid #1C3D52; }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:4px; padding:5px 14px;
  color:#6C89A0; font-size:0.75rem; font-weight:600; font-family:'Inter',sans-serif; letter-spacing:1px;
  transition: color 0.2s ease, background 0.2s ease; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#1D3A52,#152C42) !important;
  color:#2E86AB !important; border:1px solid #2E86AB !important; }

.mini-map { background:#060B12; border:1px solid #1C3D52; border-radius:6px; padding:8px;
  text-align:center; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#6C89A0; }

.safety-bar { display:flex; gap:6px; align-items:center; padding:6px 10px; background:#101B26;
  border:1px solid #1C3D52; border-radius:5px; margin-bottom:8px; }
.safety-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.safety-safe    { background:#27AE60; box-shadow:0 0 6px #27AE60; animation:breathe 2.4s ease-in-out infinite; }
.safety-caution { background:#E67E22; box-shadow:0 0 6px #E67E22; animation:pulse-red 1.4s ease-in-out infinite; }
.safety-danger  { background:#C0392B; box-shadow:0 0 6px #C0392B; animation:pulse-red 0.8s ease-in-out infinite; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.rec-dot { display:inline-block; width:9px; height:9px; background:#C0392B; border-radius:50%;
  animation:blink 0.8s ease-in-out infinite; margin-right:6px; }

.alert-crit { background:rgba(192,57,43,0.08); border-left:3px solid #C0392B; padding:6px 12px;
  margin:2px 0; border-radius:0 5px 5px 0; font-size:0.77rem; font-family:'IBM Plex Mono',monospace; }
.alert-warn { background:rgba(230,126,34,0.08); border-left:3px solid #E67E22; padding:6px 12px;
  margin:2px 0; border-radius:0 5px 5px 0; font-size:0.77rem; font-family:'IBM Plex Mono',monospace; }
.alert-info { background:rgba(46,134,171,0.06); border-left:3px solid #2E86AB; padding:6px 12px;
  margin:2px 0; border-radius:0 5px 5px 0; font-size:0.77rem; font-family:'IBM Plex Mono',monospace; }
.alert-ok   { background:rgba(39,174,96,0.06); border-left:3px solid #27AE60; padding:6px 12px;
  margin:2px 0; border-radius:0 5px 5px 0; font-size:0.77rem; font-family:'IBM Plex Mono',monospace; }

.grid-cell-btn button { min-width:38px !important; height:38px !important; padding:0 !important; }

[data-testid="stMetric"] { background:#101B26; border:1px solid #1C3D52; border-radius:6px; padding:10px; }
[data-testid="stMetricLabel"] { color:#6C89A0 !important; font-size:0.7rem !important; letter-spacing:1px; }
[data-testid="stMetricValue"] { color:#2E86AB !important; font-family:'IBM Plex Mono',monospace !important; }

.stButton > button { background:linear-gradient(135deg,#101B26,#152736) !important;
  border:1px solid #1C3D52 !important; color:#D5DEE6 !important; border-radius:5px !important;
  font-family:'Inter',sans-serif !important; font-weight:600 !important; letter-spacing:1px !important;
  font-size:0.78rem !important; text-transform:uppercase !important;
  transition: border-color 0.2s ease, color 0.2s ease, transform 0.15s ease !important; }
.stButton > button:hover { border-color:#2E86AB !important; color:#2E86AB !important; transform: translateY(-1px); }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#1D3A52,#152C42) !important;
  border-color:#2E86AB !important; color:#2E86AB !important; }

@keyframes radarPing { 0%{box-shadow:0 0 0 0 rgba(39,174,96,0.55)} 100%{box-shadow:0 0 0 8px rgba(39,174,96,0)} }
.sys-online-dot { display:inline-block; width:8px; height:8px; border-radius:50%;
  background:#27AE60; margin-right:6px; animation: radarPing 1.6s ease-out infinite; }
.sys-online-badge { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; letter-spacing:1px;
  color:#27AE60; display:inline-flex; align-items:center; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_SS_DEFAULTS: dict = {
    "tello": None, "connected": False, "flying": False, "emergency_stop": False,
    "mission_running": False, "mission_phase": "idle",
    "mission_start_time": None, "mission_id": None,
    "tel": {
        "battery": 0, "height": 0, "tof": 999,
        "speed_x": 0, "speed_y": 0, "speed_z": 0,
        "accel_x": 0.0, "accel_y": 0.0, "accel_z": 0.0,
        "pitch": 0, "roll": 0, "yaw": 0,
        "temp_lo": 0, "temp_hi": 0, "baro": 0.0, "flight_time": 0,
        "wifi_snr": "—",
    },
    "cam_active": True,
    "frame_idx": 0,
    # Path drawing (grid cells clicked in order = the ONLY path the drone will fly)
    "grid_rows": 6, "grid_cols": 6,
    "drawn_path": [],          # list of (row, col)
    "cell_size_cm": 60,        # real-world spacing per grid cell
    "path_altitude": 120,      # cm
    "path_speed": 30,          # cm/s
    "path_locked": False,
    "path_waypoints": [],
    "path_current_wp": 0,
    # Safety
    "wall_safe_dist_cm": 60,   # min obstacle distance before HOLD
    "hold_retry_limit": 4,
    "safety_min_alt": 40,
    "safety_max_alt": 400,
    "min_battery_rtl": 20,
    "auto_rtl": True,
    "obstacle_detect": True,
    "safety_status": "SAFE",
    # Video capture
    "recording": False,
    "video_frames": [],
    "last_video_path": None,
    "auto_record_on_mission": True,
    # Logs / stats
    "flight_log": [],
    "alerts": [],
    "session_stats": {"flight_distance_m": 0.0, "missions_completed": 0, "wall_holds": 0},
    "battery_ts": [],
    "zoom_level": 1.0,
    "cam_filter": "Normal",
    "screenshots": [],
    "auto_reconnect": False,
    "reconnect_count": 0,
    "stream_health": "—",
    "move_speed": 30,
    "last_fps": 0.0,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
#  ALERT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def push_alert(msg: str, level: str = "info"):
    entry = {"ts": datetime.datetime.now().strftime("%H:%M:%S"), "msg": msg, "level": level}
    st.session_state["alerts"].insert(0, entry)
    st.session_state["alerts"] = st.session_state["alerts"][:60]
    try:
        ALERT_Q.put_nowait(entry)
    except queue.Full:
        pass


def drain_alert_queue():
    while True:
        try:
            entry = ALERT_Q.get_nowait()
            if entry not in st.session_state["alerts"][:5]:
                st.session_state["alerts"].insert(0, entry)
        except queue.Empty:
            break
    st.session_state["alerts"] = st.session_state["alerts"][:60]


# ══════════════════════════════════════════════════════════════════════════════
#  SENSOR ABSTRACTION — wall / obstacle distance
# ══════════════════════════════════════════════════════════════════════════════
def get_obstacle_distance_cm() -> int:
    """
    Distance to the nearest obstacle in the direction of travel, in cm.
    Retail Tello only exposes a DOWNWARD ToF sensor. This function assumes
    Tello EDU + Mission Pad hardware (per deployment target) where mission
    pads / an add-on rangefinder can report facing distance. Swap the body
    of this function with your real sensor read if your hardware differs.
    """
    return st.session_state.get("tel", {}).get("tof", 999)


# ══════════════════════════════════════════════════════════════════════════════
#  SAFETY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_safety() -> str:
    tel = st.session_state.get("tel", {})
    dist = get_obstacle_distance_cm()
    alt  = tel.get("height", 0)
    bat  = tel.get("battery", 100)
    if not st.session_state.get("flying", False):
        return "SAFE"

    safe_dist = st.session_state.get("wall_safe_dist_cm", 60)
    max_alt   = st.session_state.get("safety_max_alt", 400)

    if dist < safe_dist or alt > max_alt or bat < 10:
        return "DANGER"
    if dist < safe_dist * 2 or alt > max_alt * 0.9 or bat < 20:
        return "CAUTION"
    return "SAFE"


def is_path_clear() -> bool:
    """True if the obstacle sensor reports enough clearance to keep moving."""
    if not st.session_state.get("obstacle_detect", True):
        return True
    return get_obstacle_distance_cm() >= st.session_state.get("wall_safe_dist_cm", 60)


def is_safe_to_move(direction: str) -> bool:
    tel = st.session_state.get("tel", {})
    alt = tel.get("height", 0)
    min_alt = st.session_state.get("safety_min_alt", 40)
    max_alt = st.session_state.get("safety_max_alt", 400)

    if direction == "fwd" and not is_path_clear():
        push_alert(f"🛡️ SAFETY: obstacle {get_obstacle_distance_cm()}cm ahead — blocked", "crit")
        return False
    if direction == "up" and alt >= max_alt:
        push_alert(f"🛡️ SAFETY: max altitude {max_alt}cm reached", "warn")
        return False
    if direction == "down" and alt <= min_alt:
        push_alert(f"🛡️ SAFETY: min altitude {min_alt}cm reached", "warn")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  PATH DRAWING  →  the exact route Auto Drive will fly, nothing else
# ══════════════════════════════════════════════════════════════════════════════
def toggle_cell(r: int, c: int):
    path = st.session_state["drawn_path"]
    if path and path[-1] == (r, c):
        return  # ignore accidental double-click of same cell
    path.append((r, c))
    st.session_state["drawn_path"] = path


def undo_last_point():
    if st.session_state["drawn_path"]:
        st.session_state["drawn_path"] = st.session_state["drawn_path"][:-1]


def clear_path():
    st.session_state["drawn_path"] = []
    st.session_state["path_waypoints"] = []
    st.session_state["path_current_wp"] = 0


def fill_preset(preset: str, rows: int, cols: int):
    cells = []
    if preset == "Grid Scan":
        for r in range(rows):
            col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in col_range:
                cells.append((r, c))
    elif preset == "Perimeter Loop":
        for c in range(cols):
            cells.append((0, c))
        for r in range(1, rows):
            cells.append((r, cols - 1))
        for c in range(cols - 2, -1, -1):
            cells.append((rows - 1, c))
        for r in range(rows - 2, 0, -1):
            cells.append((r, 0))
    elif preset == "Zigzag":
        for r in range(0, rows, 2):
            for c in range(cols):
                cells.append((r, c))
            if r + 1 < rows:
                cells.append((r + 1, cols - 1))
    st.session_state["drawn_path"] = cells


def path_to_waypoints(step_cm: int, altitude: int) -> list:
    wps = []
    for i, (r, c) in enumerate(st.session_state.get("drawn_path", [])):
        wps.append({
            "x": c * step_cm, "y": r * step_cm, "z": altitude,
            "label": f"P{i+1}(r{r},c{c})",
        })
    return wps


def path_minimap_svg(waypoints, current_wp):
    if not waypoints:
        return '<div class="mini-map" style="height:80px">No path drawn yet — draw one in AUTO DRIVE.</div>'
    W, H, margin = 260, 150, 14
    xs = [w["x"] for w in waypoints]; ys = [w["y"] for w in waypoints]
    min_x, max_x = min(xs), max(xs) + 1
    min_y, max_y = min(ys), max(ys) + 1

    def nx(x): return margin + (x - min_x) / max(max_x - min_x, 1) * (W - 2 * margin)
    def ny(y): return margin + (y - min_y) / max(max_y - min_y, 1) * (H - 2 * margin)

    lines = ""
    for i in range(1, len(waypoints)):
        x1, y1 = nx(waypoints[i-1]["x"]), ny(waypoints[i-1]["y"])
        x2, y2 = nx(waypoints[i]["x"]), ny(waypoints[i]["y"])
        lines += f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="#1C3D52" stroke-width="2"/>'

    dots = ""
    for i, wp in enumerate(waypoints):
        x, y = nx(wp["x"]), ny(wp["y"])
        color = "#27AE60" if i < current_wp else "#2E86AB" if i == current_wp else "#24404F"
        r = 5 if i == current_wp else 3
        dots += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="{color}"/>'

    return f"""
<div class="mini-map">
  <svg width="{W}" height="{H}" style="display:block;margin:0 auto">
    <rect width="{W}" height="{H}" fill="#060B12" rx="4"/>{lines}{dots}
  </svg>
  <div style="margin-top:4px;font-size:0.6rem;color:#6C89A0">
    WP {current_wp}/{len(waypoints)} · {waypoints[min(current_wp, len(waypoints)-1)]['label']}
  </div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO DRIVE MISSION — flies ONLY the drawn path. No rerouting.
# ══════════════════════════════════════════════════════════════════════════════
def _auto_drive_mission_thread():
    tello = st.session_state.get("tello")
    if tello is None:
        push_alert("No Tello connected — cannot start Auto Drive.", "crit")
        st.session_state["mission_running"] = False
        return

    step_cm  = st.session_state.get("cell_size_cm", 60)
    alt      = st.session_state.get("path_altitude", 120)
    spd      = max(10, min(100, st.session_state.get("path_speed", 30)))
    max_alt  = st.session_state.get("safety_max_alt", 400)
    min_alt  = st.session_state.get("safety_min_alt", 40)
    retry_limit = st.session_state.get("hold_retry_limit", 4)

    waypoints = path_to_waypoints(step_cm, alt)
    st.session_state["path_waypoints"] = waypoints
    st.session_state["path_current_wp"] = 0
    st.session_state["path_locked"] = True

    if not waypoints:
        push_alert("No path drawn — draw a path before starting Auto Drive.", "crit")
        st.session_state["mission_running"] = False
        st.session_state["path_locked"] = False
        return

    push_alert(f"🤖 AUTO DRIVE: {len(waypoints)} waypoints locked in — flying drawn path only", "ok")

    if st.session_state.get("auto_record_on_mission", True):
        start_recording()

    def check_abort():
        return (st.session_state.get("emergency_stop", False) or
                not st.session_state.get("mission_running", False))

    pos = {"x": 0, "y": 0}

    def move_to(dx, dy, dz):
        """Move exactly toward the next drawn waypoint. If blocked by the
        wall/obstacle sensor: HOLD in place and retry — never deviate onto
        a different path. Returns True if the move succeeded."""
        if check_abort():
            return False

        current_alt = st.session_state["tel"].get("height", 0)
        target_alt  = max(min_alt, min(max_alt, current_alt + dz))
        dz_clamped  = target_alt - current_alt

        moving_fwd = dx > 20
        attempts = 0
        while moving_fwd and not is_path_clear():
            attempts += 1
            st.session_state["mission_phase"] = "wall_hold"
            st.session_state["session_stats"]["wall_holds"] += 1
            try:
                tello.send_rc_control(0, 0, 0, 0)  # HOLD — do not reroute
            except Exception:
                pass
            push_alert(
                f"🧱 Wall/obstacle {get_obstacle_distance_cm()}cm ahead — holding "
                f"({attempts}/{retry_limit})", "warn"
            )
            if attempts >= retry_limit:
                push_alert("⛔ Path still blocked — aborting mission, landing in place.", "crit")
                return False
            time.sleep(2.0)
            if check_abort():
                return False

        st.session_state["mission_phase"] = "auto_driving"
        try:
            x_int = int(dx) if abs(int(dx)) >= 20 else 0
            y_int = int(dy) if abs(int(dy)) >= 20 else 0
            z_int = int(dz_clamped) if abs(int(dz_clamped)) >= 20 else 0
            if x_int == 0 and y_int == 0 and z_int == 0:
                return True
            tello.go_xyz_speed(y_int, x_int, z_int, spd)
            time.sleep(0.6)
            pos["x"] += dx
            pos["y"] += dy
            return True
        except Exception:
            try:
                if dx > 20:   tello.move_forward(min(500, int(dx)))
                elif dx < -20: tello.move_back(min(500, int(-dx)))
                if dy > 20:   tello.move_right(min(500, int(dy)))
                elif dy < -20: tello.move_left(min(500, int(-dy)))
                if dz_clamped > 20:   tello.move_up(min(500, int(dz_clamped)))
                elif dz_clamped < -20: tello.move_down(min(500, int(-dz_clamped)))
                pos["x"] += dx; pos["y"] += dy
                return True
            except Exception as e:
                push_alert(f"Move error: {e}", "warn")
                return False

    try:
        st.session_state["mission_phase"] = "takeoff"
        tello.takeoff()
        time.sleep(3)

        current_h = st.session_state["tel"].get("height", 0)
        delta_h = min(max_alt, alt) - current_h
        if delta_h > 20:
            move_to(0, 0, delta_h)
            time.sleep(1.5)

        st.session_state["mission_phase"] = "auto_driving"
        push_alert("📡 Auto Drive engaged — following drawn path", "ok")

        prev_x, prev_y = 0, 0
        for i, wp in enumerate(waypoints):
            if check_abort():
                break
            st.session_state["path_current_wp"] = i

            bat = st.session_state["tel"].get("battery", 100)
            if bat <= st.session_state.get("min_battery_rtl", 20):
                push_alert(f"🔋 Battery critical ({bat}%) — landing.", "crit")
                break

            dx = wp["x"] - prev_x
            dy = wp["y"] - prev_y
            dz = wp["z"] - st.session_state["tel"].get("height", alt)

            moved = move_to(dx, dy, dz)
            if not moved:
                break  # blocked path — abort, do not skip/reroute

            time.sleep(1.2)
            prev_x, prev_y = wp["x"], wp["y"]
            push_alert(f"✅ WP {i+1}/{len(waypoints)}: {wp['label']}", "info")

        st.session_state["path_current_wp"] = len(waypoints)

    except Exception as e:
        push_alert(f"Auto Drive error: {e}", "crit")

    finally:
        st.session_state["mission_phase"] = "landing"
        push_alert("🛬 Auto Drive complete — landing…", "warn")
        try:
            tello.land()
        except Exception:
            pass
        st.session_state["flying"] = False
        st.session_state["mission_running"] = False
        st.session_state["mission_phase"] = "idle"
        st.session_state["path_locked"] = False
        st.session_state["session_stats"]["missions_completed"] += 1
        if st.session_state.get("recording"):
            stop_recording()
            save_video_path = build_mp4_from_frames()
            if save_video_path:
                push_alert(f"🎬 Flight video saved → {save_video_path}", "ok")
        push_alert("✅ Auto Drive mission complete.", "ok")


def start_auto_drive():
    if not st.session_state.get("flying") and not st.session_state.get("connected"):
        push_alert("Connect the drone first.", "warn"); return
    if not st.session_state.get("drawn_path"):
        push_alert("Draw a path first (AUTO DRIVE tab).", "warn"); return
    st.session_state.update({
        "mission_running": True, "emergency_stop": False,
        "mission_id": str(uuid.uuid4())[:8], "path_current_wp": 0,
    })
    push_alert("🚀 Launching Auto Drive…", "ok")
    threading.Thread(target=_auto_drive_mission_thread, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA FILTER & ZOOM
# ══════════════════════════════════════════════════════════════════════════════
def apply_cam_filter(bgr: np.ndarray, filter_name: str, zoom: float) -> np.ndarray:
    if not CV2_AVAILABLE:
        return bgr
    if zoom > 1.01:
        h, w = bgr.shape[:2]
        cx, cy = w // 2, h // 2
        crop_w = max(64, int(w / zoom)); crop_h = max(48, int(h / zoom))
        x1 = max(0, min(cx - crop_w // 2, w - crop_w))
        y1 = max(0, min(cy - crop_h // 2, h - crop_h))
        bgr = cv2.resize(bgr[y1:y1+crop_h, x1:x1+crop_w], (w, h), interpolation=cv2.INTER_LINEAR)

    if filter_name == "Grayscale":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_name == "Edge Detection":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        edges = cv2.Canny(eq, 40, 120)
        bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_name == "Night Vision":
        boosted = cv2.convertScaleAbs(bgr, alpha=2.0, beta=40)
        green = np.zeros_like(boosted)
        green[:, :, 1] = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
        bgr = green
    elif filter_name == "Thermal":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return bgr


def draw_hud(bgr, tel, phase, fidx, fps=0, zoom=1.0, filt="Normal", recording=False,
             safety="SAFE", wp_progress=""):
    if not CV2_AVAILABLE:
        return bgr
    h, w = bgr.shape[:2]
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, h-48), (w, h), (0, 0, 0), -1)
    bgr = cv2.addWeighted(overlay, 0.6, bgr, 0.4, 0)

    bat = tel.get("battery", 0)
    bat_color = (0,255,136) if bat>50 else (0,165,255) if bat>20 else (0,71,255)
    cv2.putText(bgr, f"BAT:{bat}%", (8, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bat_color, 1, cv2.LINE_AA)
    cv2.putText(bgr, f"ALT:{tel.get('height',0)}cm", (90, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,212,255), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"YAW:{tel.get('yaw',0):.0f}deg", (200, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,212,255), 1, cv2.LINE_AA)
    dist = tel.get("tof", 0)
    dist_color = (0,255,136) if dist>120 else (0,165,255) if dist>60 else (0,71,255)
    cv2.putText(bgr, f"WALL:{dist}cm", (310, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dist_color, 1, cv2.LINE_AA)
    cv2.putText(bgr, f"FPS:{fps:.0f}", (420, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

    phase_color = (0,255,136) if phase=="auto_driving" else (0,71,255) if phase=="wall_hold" else (200,200,200)
    cv2.putText(bgr, phase.upper(), (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, phase_color, 1, cv2.LINE_AA)

    safety_color = (0,255,136) if safety=="SAFE" else (0,165,255) if safety=="CAUTION" else (0,71,255)
    cv2.putText(bgr, f"SAFETY:{safety}", (w-170, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, safety_color, 1, cv2.LINE_AA)

    if wp_progress:
        cv2.putText(bgr, wp_progress, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,212,255), 1, cv2.LINE_AA)
    if zoom > 1.01:
        cv2.putText(bgr, f"ZOOM {zoom:.1f}x", (w-120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,220,0), 1, cv2.LINE_AA)
    if filt != "Normal":
        cv2.putText(bgr, filt.upper(), (w-160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,255,220), 1, cv2.LINE_AA)
    if recording:
        cv2.circle(bgr, (w-15, 15), 6, (0,71,255), -1)
    cv2.putText(bgr, f"#{fidx}", (w-40, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1, cv2.LINE_AA)
    return bgr


# ══════════════════════════════════════════════════════════════════════════════
#  MJPEG SERVER (local, for the true-live in-browser feed)
# ══════════════════════════════════════════════════════════════════════════════
def _mjpeg_server():
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json as _json

    class MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass

        def do_GET(self):
            if self.path == "/video_feed":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                while True:
                    with _MJPEG_LOCK:
                        frame = bytes(_MJPEG_FRAME)
                    if frame:
                        try:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                        except Exception:
                            break
                    time.sleep(0.04)
            elif self.path == "/snapshot":
                with _MJPEG_LOCK:
                    frame = bytes(_MJPEG_FRAME)
                if frame:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(frame)
                else:
                    self.send_response(204); self.end_headers()
            elif self.path == "/meta":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with _MJPEG_LOCK:
                    meta = dict(_MJPEG_META)
                self.wfile.write(_json.dumps(meta).encode())
            else:
                self.send_response(404); self.end_headers()

    srv = HTTPServer(("0.0.0.0", _MJPEG_PORT), MJPEGHandler)
    srv.serve_forever()


def _ensure_mjpeg_server():
    global _MJPEG_STARTED
    if not _MJPEG_STARTED:
        threading.Thread(target=_mjpeg_server, daemon=True).start()
        _MJPEG_STARTED = True
        time.sleep(0.2)


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA THREAD  (live view + HUD + optional recording — no defect detection)
# ══════════════════════════════════════════════════════════════════════════════
def _camera_thread():
    global _MJPEG_FRAME, _MJPEG_META
    tello = st.session_state.get("tello")
    if tello is None:
        push_alert("Camera thread: no Tello object.", "crit")
        return
    try:
        tello.streamon()
        time.sleep(0.7)
        reader = tello.get_frame_read()
    except Exception as e:
        push_alert(f"Stream failed: {e}", "crit")
        st.session_state["cam_active"] = False
        return

    push_alert("📷 Live camera stream active", "ok")
    st.session_state["stream_health"] = "OK"
    _frame_times = []
    _stall_count = 0

    while st.session_state.get("cam_active", False):
        try:
            raw = reader.frame
            if raw is None or raw.size == 0:
                _stall_count += 1
                if _stall_count > 30:
                    st.session_state["stream_health"] = "WARN"
                time.sleep(0.03); continue
            _stall_count = 0
            st.session_state["stream_health"] = "OK"

            bgr = cv2.resize(raw.copy(), (854, 480))
            filt = st.session_state.get("cam_filter", "Normal")
            zoom = st.session_state.get("zoom_level", 1.0)
            ann = apply_cam_filter(bgr, filt, zoom)

            now_t = time.time()
            _frame_times.append(now_t)
            _frame_times = [t for t in _frame_times if now_t - t < 2.0]
            fps = len(_frame_times) / 2.0
            st.session_state["last_fps"] = round(fps, 1)

            tel_snap = dict(st.session_state.get("tel", {}))
            phase = st.session_state.get("mission_phase", "idle")
            fidx = st.session_state.get("frame_idx", 0)
            rec = st.session_state.get("recording", False)
            safety = evaluate_safety()

            wps = st.session_state.get("path_waypoints", [])
            cur = st.session_state.get("path_current_wp", 0)
            wp_progress = f"WP {cur}/{len(wps)}" if wps else ""

            ann = draw_hud(ann, tel_snap, phase, fidx, fps=fps, zoom=zoom, filt=filt,
                            recording=rec, safety=safety, wp_progress=wp_progress)

            bat_now = tel_snap.get("battery", 0)
            if bat_now > 0:
                hist = st.session_state.get("battery_ts", [])
                hist.append((now_t, bat_now))
                if len(hist) > 120: hist = hist[-120:]
                st.session_state["battery_ts"] = hist

            ok, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                jpeg_bytes = buf.tobytes()
                with _MJPEG_LOCK:
                    _MJPEG_FRAME = jpeg_bytes
                    _MJPEG_META = {"tel": tel_snap, "frame_idx": fidx}
                if rec:
                    vf = st.session_state.get("video_frames", [])
                    vf.append(jpeg_bytes)
                    if len(vf) > 3000: vf = vf[-3000:]
                    st.session_state["video_frames"] = vf

            st.session_state["frame_idx"] += 1

        except Exception as e:
            push_alert(f"Camera error: {e}", "warn")
            st.session_state["stream_health"] = "ERROR"
        time.sleep(0.04)

    try:
        tello.streamoff()
    except Exception:
        pass
    with _MJPEG_LOCK:
        _MJPEG_FRAME = b""
    st.session_state["stream_health"] = "—"
    push_alert("📷 Camera stopped.", "info")


def start_camera():
    if not CV2_AVAILABLE:
        push_alert("opencv-python not installed.", "crit"); return
    if st.session_state.get("cam_active") and st.session_state.get("_cam_thread_started"):
        return
    _ensure_mjpeg_server()
    st.session_state["cam_active"] = True
    st.session_state["_cam_thread_started"] = True
    threading.Thread(target=_camera_thread, daemon=True).start()


def stop_camera():
    st.session_state["cam_active"] = False
    st.session_state["_cam_thread_started"] = False
    st.session_state["recording"] = False


def capture_screenshot():
    with _MJPEG_LOCK:
        frame = bytes(_MJPEG_FRAME)
    if not frame:
        push_alert("No frame to capture.", "warn"); return
    b64 = base64.b64encode(frame).decode()
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    shots = st.session_state.get("screenshots", [])
    shots.insert(0, {"id": str(uuid.uuid4())[:8], "ts": ts, "b64": b64})
    st.session_state["screenshots"] = shots[:50]
    push_alert(f"📸 Screenshot saved {ts}", "ok")


def battery_eta() -> str:
    hist = st.session_state.get("battery_ts", [])
    if len(hist) < 6: return "—"
    t0, b0 = hist[0]; t1, b1 = hist[-1]
    elapsed = t1 - t0
    if elapsed < 5 or b0 <= b1: return "—"
    drain_per_sec = (b0 - b1) / elapsed
    if drain_per_sec <= 0: return "—"
    seconds_left = b1 / drain_per_sec
    m, s = int(seconds_left // 60), int(seconds_left % 60)
    return f"{m}:{s:02d}"


def start_recording():
    st.session_state["recording"] = True
    st.session_state["video_frames"] = []
    push_alert("🔴 Video capture started", "ok")


def stop_recording():
    st.session_state["recording"] = False
    push_alert(f"⏹️ Video capture stopped — {len(st.session_state.get('video_frames',[]))} frames", "info")


def build_mp4_from_frames() -> str | None:
    """Encode captured JPEG frames into a real .mp4 file on disk."""
    frames = st.session_state.get("video_frames", [])
    if not frames or not CV2_AVAILABLE:
        return None
    first = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        return None
    h, w = first.shape[:2]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(VIDEO_DIR, f"auto_drive_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
    for jpeg in frames:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            writer.write(cv2.resize(img, (w, h)))
    writer.release()
    st.session_state["last_video_path"] = path
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  TELEMETRY THREAD — polls every available Tello sensor
# ══════════════════════════════════════════════════════════════════════════════
def _telemetry_thread():
    tello = st.session_state.get("tello")
    if tello is None: return
    while st.session_state.get("connected"):
        if st.session_state.get("emergency_stop"): break
        try:
            tel = st.session_state["tel"]
            try: tel["battery"] = tello.get_battery()
            except Exception: pass
            try: tel["tof"] = tello.get_distance_tof()
            except Exception: pass

            flying = st.session_state.get("flying", False)
            if flying:
                for key, fn in [
                    ("height", tello.get_height), ("speed_x", tello.get_speed_x),
                    ("speed_y", tello.get_speed_y), ("speed_z", tello.get_speed_z),
                    ("accel_x", tello.get_acceleration_x), ("accel_y", tello.get_acceleration_y),
                    ("accel_z", tello.get_acceleration_z), ("pitch", tello.get_pitch),
                    ("roll", tello.get_roll), ("yaw", tello.get_yaw),
                    ("temp_lo", tello.get_lowest_temperature), ("temp_hi", tello.get_highest_temperature),
                    ("baro", tello.get_barometer), ("flight_time", tello.get_flight_time),
                ]:
                    try: tel[key] = fn()
                    except Exception: pass
                try: tel["wifi_snr"] = tello.query_wifi_signal_noise_ratio()
                except Exception: pass

                st.session_state["safety_status"] = evaluate_safety()

                entry = {
                    "time": datetime.datetime.now().isoformat(),
                    "battery": tel["battery"], "height": tel["height"],
                    "yaw": tel["yaw"], "speed_x": tel["speed_x"], "speed_y": tel["speed_y"],
                    "tof": tel["tof"],
                }
                st.session_state["flight_log"].append(entry)
                if len(st.session_state["flight_log"]) > 3000:
                    st.session_state["flight_log"] = st.session_state["flight_log"][-2000:]

                spd = math.hypot(tel["speed_x"], tel["speed_y"])
                st.session_state["session_stats"]["flight_distance_m"] += spd / 100.0

            bat = tel["battery"]
            if flying and 0 < bat <= st.session_state.get("min_battery_rtl", 20):
                push_alert(f"🔋 Battery {bat}% — landing!", "crit")
                if st.session_state.get("auto_rtl", True):
                    _do_land(); break

        except Exception as e:
            push_alert(f"Telemetry: {e}", "warn")
        time.sleep(1)


# ══════════════════════════════════════════════════════════════════════════════
#  FLIGHT COMMAND HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _do_connect():
    if not TELLO_AVAILABLE:
        push_alert("djitellopy not installed. pip install djitellopy", "crit"); return False
    if not CV2_AVAILABLE:
        push_alert("opencv-python not installed. pip install opencv-python", "crit"); return False
    try:
        t = Tello()
        t.connect()
        bat = t.get_battery()
        st.session_state["tello"] = t
        st.session_state["connected"] = True
        st.session_state["tel"]["battery"] = bat
        st.session_state["reconnect_count"] += 1
        push_alert(f"✅ Tello connected. Battery: {bat}%", "ok")
        threading.Thread(target=_telemetry_thread, daemon=True).start()
        return True
    except Exception as e:
        push_alert(f"Connection failed: {e}", "crit"); return False


def _do_disconnect():
    stop_camera()
    t = st.session_state.get("tello")
    if t:
        try: t.end()
        except Exception: pass
    st.session_state.update({"tello": None, "connected": False, "flying": False,
                              "mission_running": False, "mission_phase": "idle"})
    push_alert("Tello disconnected.", "info")


def _do_takeoff():
    t = st.session_state.get("tello")
    if t is None:
        push_alert("Not connected.", "crit"); return
    try:
        t.takeoff()
        st.session_state["flying"] = True
        st.session_state["emergency_stop"] = False
        st.session_state["mission_start_time"] = datetime.datetime.now()
        st.session_state["battery_ts"] = []
        push_alert("🛫 Takeoff!", "ok")
    except Exception as e:
        push_alert(f"Takeoff failed: {e}", "crit")


def _do_land():
    t = st.session_state.get("tello")
    if t:
        try: t.land()
        except Exception: pass
    st.session_state.update({"flying": False, "mission_running": False, "mission_phase": "idle"})
    push_alert("🛬 Landed.", "ok")


def _do_emergency():
    t = st.session_state.get("tello")
    if t:
        try: t.emergency()
        except Exception: pass
    stop_camera()
    st.session_state.update({"emergency_stop": True, "flying": False,
                              "mission_running": False, "mission_phase": "idle"})
    push_alert("🚨 EMERGENCY STOP — MOTORS CUT!", "crit")


def _do_move(direction, dist=None):
    t = st.session_state.get("tello")
    spd = dist or st.session_state.get("move_speed", 30)
    if t is None: return
    if not is_safe_to_move(direction):
        return
    try:
        {
            "up":    lambda: t.move_up(spd), "down": lambda: t.move_down(spd),
            "fwd":   lambda: t.move_forward(spd), "back": lambda: t.move_back(spd),
            "left":  lambda: t.move_left(spd), "right": lambda: t.move_right(spd),
            "cw":    lambda: t.rotate_clockwise(45), "ccw": lambda: t.rotate_counter_clockwise(45),
        }[direction]()
        push_alert(f"↗️ {direction.upper()} {spd}cm", "info")
    except Exception as e:
        push_alert(f"Move error: {e}", "warn")


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def kpi(val, label, sub="", color="#2E86AB"):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi-card"><div class="kpi-val" style="color:{color}">{val}</div>'
            f'<div class="kpi-lbl">{label}</div>{sub_html}</div>')


def pill(text, kind="off"):
    return f'<span class="s-pill s-{kind}">{text}</span>'


def mission_elapsed():
    t0 = st.session_state.get("mission_start_time")
    if not t0: return "--:--:--"
    d = datetime.datetime.now() - t0
    m, s = divmod(int(d.total_seconds()), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
drain_alert_queue()

if st.session_state.get("cam_active"):
    if AUTOREFRESH_AVAILABLE:
        _st_autorefresh(interval=150, key="cam_refresh")

tel     = st.session_state["tel"]
stats   = st.session_state["session_stats"]
conn    = st.session_state["connected"]
flying  = st.session_state["flying"]
mission = st.session_state["mission_running"]
phase   = st.session_state["mission_phase"]
bat     = tel.get("battery", 0)
safety  = st.session_state.get("safety_status", "SAFE")

st.markdown(f"""
<div class="hud-header">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div class="hud-title">🚁 DRONE CONTROL CENTER</div>
      <div class="hud-subtitle">Tello Auto Drive · draw a path · fly that path only · keep clear of walls · capture video</div>
    </div>
    <div style="text-align:right">
      <div class="sys-online-badge"><span class="sys-online-dot"></span>SYSTEM ONLINE</div>
      <div class="hud-version">SYS: {datetime.datetime.now().strftime('%H:%M:%S')}</div>
      <div class="hud-version" style="color:{'#27AE60' if safety=='SAFE' else '#E67E22' if safety=='CAUTION' else '#C0392B'}">
        SAFETY: {safety}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

sb = st.columns(7)
sb[0].markdown(pill("CONNECTED" if conn else "OFFLINE", "on" if conn else "off"), unsafe_allow_html=True)
sb[1].markdown(pill("FLYING" if flying else "GROUNDED", "on" if flying else "off"), unsafe_allow_html=True)
sb[2].markdown(pill(f"{phase.upper()}" if mission else "IDLE", "on" if mission else "off"), unsafe_allow_html=True)
sb[3].markdown(pill(f"BAT {bat:.0f}%", "on" if bat>50 else "warn" if bat>20 else "crit"), unsafe_allow_html=True)
sb[4].markdown(pill(f"ALT {tel.get('height',0)}cm", "on" if flying else "off"), unsafe_allow_html=True)
_d = tel.get('tof', 0)
sb[5].markdown(pill(f"WALL {_d}cm", "on" if _d>120 else "warn" if _d>60 else "crit"), unsafe_allow_html=True)
_health = st.session_state.get("stream_health", "—")
sb[6].markdown(pill(f"STREAM {_health}", {"OK":"on","WARN":"warn","ERROR":"crit"}.get(_health,"off")), unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1C3D52;margin:5px 0 8px'>", unsafe_allow_html=True)

tab_ctrl, tab_cam, tab_auto, tab_telem = st.tabs([
    "🎮 CONTROL", "📷 LIVE CAMERA", "🧭 AUTO DRIVE", "📡 TELEMETRY",
])

# ════════════════════════════════════════════════════════════════════════════
#  TAB — CONTROL
# ════════════════════════════════════════════════════════════════════════════
with tab_ctrl:
    ctrl_left, ctrl_mid, ctrl_right = st.columns([1, 1, 1], gap="medium")

    with ctrl_left:
        st.markdown('<div class="sec-hdr">📶 CONNECTION</div>', unsafe_allow_html=True)
        if not conn:
            st.markdown("""
<div style="background:#101B26;border:1px solid #1C3D52;border-radius:6px;padding:10px 14px;font-size:0.78rem;margin-bottom:8px">
  <b style="color:#2E86AB">Setup:</b><br>
  1. Power on Tello (LED blinks yellow)<br>
  2. Connect this device to <code style="color:#2E86AB">TELLO-XXXXXX</code> WiFi<br>
  3. Click Connect ↓
</div>""", unsafe_allow_html=True)
            st.session_state["auto_reconnect"] = st.toggle(
                "Auto-reconnect on drop", value=st.session_state["auto_reconnect"], key="ar_toggle")
            if st.button("🔗  CONNECT TO TELLO", key="btn_connect", use_container_width=True, type="primary"):
                if _do_connect(): st.rerun()
        else:
            st.success(f"🟢 Online — {bat:.0f}% bat | ETA: {battery_eta()}")
            st.progress(int(bat) / 100)
            c1, c2 = st.columns(2)
            c1.metric("Battery", f"{bat:.0f}%")
            c2.metric("ETA", battery_eta())
            if st.button("🔌 Disconnect", key="btn_disconnect", use_container_width=True):
                _do_disconnect(); st.rerun()

        st.markdown('<div class="sec-hdr">✈️ FLIGHT</div>', unsafe_allow_html=True)
        fb1, fb2 = st.columns(2)
        with fb1:
            if st.button("🛫 TAKEOFF", key="btn_takeoff", use_container_width=True,
                         type="primary", disabled=not conn or flying):
                _do_takeoff(); st.rerun()
            if st.button("🏠 LAND NOW", key="btn_rtl", use_container_width=True, disabled=not flying):
                _do_land(); st.rerun()
        with fb2:
            if st.button("🛬 LAND", key="btn_land", use_container_width=True, disabled=not flying):
                _do_land(); st.rerun()
            if st.button("🚨 E-STOP", key="btn_estop", use_container_width=True, disabled=not conn):
                _do_emergency(); st.rerun()

        st.markdown('<div class="sec-hdr">📷 CAMERA</div>', unsafe_allow_html=True)
        if not st.session_state["cam_active"]:
            if st.button("▶️ START CAMERA", key="btn_cam_on", use_container_width=True,
                         type="primary", disabled=not conn):
                start_camera(); st.rerun()
        else:
            if st.button("⏹️ STOP CAMERA", key="btn_cam_off", use_container_width=True):
                stop_camera(); st.rerun()

        if st.button("📸 Screenshot", key="btn_snap", use_container_width=True,
                     disabled=not st.session_state["cam_active"]):
            capture_screenshot(); st.rerun()

        if not st.session_state.get("recording"):
            if st.button("🔴 Start Recording", key="btn_rec", use_container_width=True,
                         disabled=not st.session_state["cam_active"]):
                start_recording(); st.rerun()
        else:
            st.markdown('<span class="rec-dot"></span>**RECORDING…**', unsafe_allow_html=True)
            if st.button("⏹️ Stop & Save Video", key="btn_rec_stop", use_container_width=True):
                stop_recording()
                p = build_mp4_from_frames()
                if p: st.success(f"Saved: {p}")
                st.rerun()

        if st.session_state.get("last_video_path") and os.path.exists(st.session_state["last_video_path"]):
            with open(st.session_state["last_video_path"], "rb") as f:
                st.download_button("⬇️ Download Last Recording", data=f.read(),
                                   file_name=os.path.basename(st.session_state["last_video_path"]),
                                   mime="video/mp4", key="dl_vid_ctrl", use_container_width=True)

    with ctrl_mid:
        st.markdown('<div class="sec-hdr">🕹️ MANUAL CONTROLS</div>', unsafe_allow_html=True)
        dis = not flying or mission
        speed_val = st.select_slider("Move Speed (cm)", options=[20, 30, 40, 50, 80, 100],
                                      value=st.session_state.get("move_speed", 30), key="spd_slider")
        st.session_state["move_speed"] = speed_val

        r1 = st.columns(4)
        with r1[0]:
            if st.button("▲ FWD", key="btn_fwd", use_container_width=True, disabled=dis): _do_move("fwd")
        with r1[1]:
            if st.button("▼ BACK", key="btn_back", use_container_width=True, disabled=dis): _do_move("back")
        with r1[2]:
            if st.button("◀ LEFT", key="btn_left", use_container_width=True, disabled=dis): _do_move("left")
        with r1[3]:
            if st.button("▶ RIGHT", key="btn_right", use_container_width=True, disabled=dis): _do_move("right")

        r2 = st.columns(4)
        with r2[0]:
            if st.button("⬆ UP", key="btn_up", use_container_width=True, disabled=dis): _do_move("up")
        with r2[1]:
            if st.button("⬇ DOWN", key="btn_down", use_container_width=True, disabled=dis): _do_move("down")
        with r2[2]:
            if st.button("↺ CCW", key="btn_ccw", use_container_width=True, disabled=dis): _do_move("ccw")
        with r2[3]:
            if st.button("↻ CW", key="btn_cw", use_container_width=True, disabled=dis): _do_move("cw")

        st.caption("Manual controls lock while Auto Drive is running.")

    with ctrl_right:
        st.markdown('<div class="sec-hdr">📷 LIVE FEED</div>', unsafe_allow_html=True)
        if st.session_state.get("cam_active"):
            live_camera_component(height=260)
            st.caption(f"FPS: {st.session_state.get('last_fps',0):.1f} · Frame #{st.session_state['frame_idx']}")
        else:
            st.markdown("""
<div class="cam-panel"><div class="cam-offline">
  <div style="font-size:1.5rem">📷</div><div>CAMERA OFFLINE</div>
  <div style="color:#24404F">Connect drone &amp; start camera</div>
</div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">🎛️ CAMERA SETTINGS</div>', unsafe_allow_html=True)
        z_col, f_col = st.columns(2)
        with z_col:
            st.session_state["zoom_level"] = st.slider("Zoom", 1.0, 4.0, st.session_state["zoom_level"], 0.25, key="z_ctrl")
        with f_col:
            st.session_state["cam_filter"] = st.selectbox(
                "Filter", CAM_FILTERS, index=CAM_FILTERS.index(st.session_state["cam_filter"]), key="f_ctrl")

# ════════════════════════════════════════════════════════════════════════════
#  TAB — LIVE CAMERA
# ════════════════════════════════════════════════════════════════════════════
with tab_cam:
    cam_l, cam_r = st.columns([3, 1], gap="medium")

    with cam_r:
        st.markdown('<div class="sec-hdr">🎛️ STREAM CONTROLS</div>', unsafe_allow_html=True)
        if not st.session_state["cam_active"]:
            if st.button("▶️ Start Camera", key="btn_cam2_on", use_container_width=True,
                         type="primary", disabled=not conn):
                start_camera(); st.rerun()
        else:
            if st.button("⏹️ Stop Camera", key="btn_cam2_off", use_container_width=True):
                stop_camera(); st.rerun()

        if st.button("📸 Capture", key="btn_snap2", use_container_width=True,
                     disabled=not st.session_state["cam_active"]):
            capture_screenshot(); st.rerun()
        if st.button("🔄 Refresh", key="btn_refresh2", use_container_width=True):
            st.rerun()

        st.markdown('<div class="sec-hdr">🔍 ZOOM</div>', unsafe_allow_html=True)
        st.session_state["zoom_level"] = st.slider("Zoom", 1.0, 4.0, st.session_state["zoom_level"], 0.25, key="zoom_sl2")

        st.markdown('<div class="sec-hdr">🎨 FILTER</div>', unsafe_allow_html=True)
        st.session_state["cam_filter"] = st.selectbox(
            "Filter", CAM_FILTERS, index=CAM_FILTERS.index(st.session_state["cam_filter"]), key="filt_sl2")

        st.markdown('<div class="sec-hdr">📊 STATS</div>', unsafe_allow_html=True)
        st.metric("FPS", f"{st.session_state.get('last_fps',0):.1f}")
        st.metric("Frame #", st.session_state["frame_idx"])
        st.metric("Screenshots", len(st.session_state.get("screenshots", [])))

    with cam_l:
        if not st.session_state["cam_active"]:
            st.info("📷 Camera is off. Connect Tello and press ▶️ Start Camera.")
            st.markdown("""
<div class="cam-panel"><div class="cam-offline">
  <div style="font-size:2rem">📷</div><div>CAMERA OFFLINE</div>
  <div style="color:#24404F;font-size:0.75rem">Connect drone &amp; start camera</div>
</div></div>""", unsafe_allow_html=True)
        else:
            live_camera_component(height=440)
            extras = []
            if st.session_state["zoom_level"] > 1.0: extras.append(f"🔭 {st.session_state['zoom_level']:.1f}×")
            if st.session_state["cam_filter"] != "Normal": extras.append(f"🎨 {st.session_state['cam_filter']}")
            if st.session_state.get("recording"): extras.append("🔴 REC")
            st.caption(f"🟢 Live · Frame #{st.session_state['frame_idx']} · FPS {st.session_state.get('last_fps',0):.1f}"
                       + (" · " + " · ".join(extras) if extras else ""))

# ════════════════════════════════════════════════════════════════════════════
#  TAB — AUTO DRIVE  (draw a path → drone flies exactly that path)
# ════════════════════════════════════════════════════════════════════════════
with tab_auto:
    auto_l, auto_r = st.columns([2, 1], gap="medium")

    with auto_l:
        st.markdown('<div class="sec-hdr">🧭 DRAW THE FLIGHT PATH</div>', unsafe_allow_html=True)
        st.caption("Click grid cells in the order you want the drone to visit them. "
                   "Auto Drive will fly exactly this sequence — if a wall/obstacle blocks "
                   "the way it holds and retries, it never improvises a different route.")

        locked = st.session_state.get("path_locked", False)
        gp1, gp2, gp3 = st.columns(3)
        with gp1:
            st.session_state["grid_rows"] = st.number_input("Grid rows", 2, 10,
                                            st.session_state["grid_rows"], key="gr_rows", disabled=locked)
        with gp2:
            st.session_state["grid_cols"] = st.number_input("Grid cols", 2, 10,
                                            st.session_state["grid_cols"], key="gr_cols", disabled=locked)
        with gp3:
            st.session_state["cell_size_cm"] = st.number_input("Cell size (cm)", 20, 200,
                                            st.session_state["cell_size_cm"], key="gr_cell", disabled=locked)

        rows, cols = st.session_state["grid_rows"], st.session_state["grid_cols"]
        path = st.session_state["drawn_path"]
        order = {cell: i + 1 for i, cell in enumerate(path)}

        for r in range(rows):
            row_cols = st.columns(cols)
            for c in range(cols):
                label = str(order[(r, c)]) if (r, c) in order else "·"
                with row_cols[c]:
                    st.markdown('<div class="grid-cell-btn">', unsafe_allow_html=True)
                    if st.button(label, key=f"cell_{r}_{c}", disabled=locked):
                        toggle_cell(r, c); st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

        gb1, gb2, gb3, gb4 = st.columns(4)
        with gb1:
            if st.button("↩️ Undo point", use_container_width=True, disabled=locked):
                undo_last_point(); st.rerun()
        with gb2:
            if st.button("🗑️ Clear path", use_container_width=True, disabled=locked):
                clear_path(); st.rerun()
        with gb3:
            preset = st.selectbox("Quick-fill preset", ["—"] + PATH_PRESETS[1:], key="preset_sel", label_visibility="collapsed")
        with gb4:
            if st.button("✨ Apply preset", use_container_width=True, disabled=locked or preset == "—"):
                fill_preset(preset, rows, cols); st.rerun()

        st.markdown('<div class="sec-hdr">⚙️ MISSION PARAMETERS</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            st.session_state["path_altitude"] = st.number_input("Altitude (cm)", 50, 400,
                                                st.session_state["path_altitude"], key="alt_auto", disabled=locked)
        with p2:
            st.session_state["path_speed"] = st.slider("Speed (cm/s)", 10, 100,
                                                st.session_state["path_speed"], key="spd_auto", disabled=locked)

        st.session_state["auto_record_on_mission"] = st.toggle(
            "🎥 Auto-capture video during Auto Drive", value=st.session_state["auto_record_on_mission"], key="rec_tog_auto")

        wps = path_to_waypoints(st.session_state["cell_size_cm"], st.session_state["path_altitude"])
        cur_wp = st.session_state.get("path_current_wp", 0)
        st.markdown(path_minimap_svg(wps, cur_wp), unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">🚀 MISSION CONTROL</div>', unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            if st.button("🚀 START AUTO DRIVE", key="btn_auto_start", use_container_width=True,
                         type="primary", disabled=not conn or mission or not path):
                start_auto_drive(); st.rerun()
        with mc2:
            if st.button("⬛ ABORT & LAND", key="btn_auto_abort", use_container_width=True, disabled=not mission):
                st.session_state["mission_running"] = False
                push_alert("Auto Drive aborted by operator.", "warn"); st.rerun()

        if mission:
            prog = cur_wp / max(len(wps), 1)
            st.progress(min(prog, 1.0))
            st.caption(f"Progress: {cur_wp}/{len(wps)} waypoints · Phase: {phase.upper()}")

        if st.session_state.get("last_video_path") and os.path.exists(st.session_state["last_video_path"]) and not mission:
            with open(st.session_state["last_video_path"], "rb") as f:
                st.download_button("⬇️ Download Mission Video", data=f.read(),
                                   file_name=os.path.basename(st.session_state["last_video_path"]),
                                   mime="video/mp4", key="dl_vid_auto", use_container_width=True)

    with auto_r:
        st.markdown('<div class="sec-hdr">🛡️ WALL / OBSTACLE SAFETY</div>', unsafe_allow_html=True)
        st.session_state["obstacle_detect"] = st.toggle(
            "Wall-distance keeping", value=st.session_state["obstacle_detect"], key="obs_tog")
        st.session_state["wall_safe_dist_cm"] = st.slider(
            "Min safe distance (cm)", 20, 200, st.session_state["wall_safe_dist_cm"], key="wall_safe")
        st.session_state["hold_retry_limit"] = st.slider(
            "Hold retries before abort", 1, 10, st.session_state["hold_retry_limit"], key="hold_retry")
        st.session_state["safety_min_alt"] = st.slider(
            "Min altitude (cm)", 20, 150, st.session_state["safety_min_alt"], key="min_alt")
        st.session_state["safety_max_alt"] = st.slider(
            "Max altitude (cm)", 100, 500, st.session_state["safety_max_alt"], key="max_alt")
        st.session_state["min_battery_rtl"] = st.slider(
            "Land-now battery %", 5, 40, st.session_state["min_battery_rtl"], key="rtl_bat")
        st.session_state["auto_rtl"] = st.toggle(
            "Auto-land on low battery", value=st.session_state["auto_rtl"], key="auto_rtl_tog")

        st.markdown('<div class="sec-hdr">📊 MISSION STATUS</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:#101B26;border:1px solid #1C3D52;border-radius:6px;padding:12px;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;line-height:1.8">
  <div>Phase: <span style="color:#2E86AB">{phase.upper()}</span></div>
  <div>Waypoints: <span style="color:#2E86AB">{cur_wp}/{len(wps)}</span></div>
  <div>Missions done: <span style="color:#27AE60">{stats['missions_completed']}</span></div>
  <div>Wall holds: <span style="color:#E67E22">{stats['wall_holds']}</span></div>
  <div>Elapsed: <span style="color:#D5DEE6">{mission_elapsed()}</span></div>
  <div>ID: <span style="color:#6C89A0">{st.session_state.get('mission_id','—')}</span></div>
</div>
""", unsafe_allow_html=True)

        safety_now = evaluate_safety()
        s_dot = "safety-safe" if safety_now == "SAFE" else "safety-caution" if safety_now == "CAUTION" else "safety-danger"
        st.markdown(f"""
<div class="safety-bar" style="margin-top:8px">
  <div class="safety-dot {s_dot}"></div>
  <div style="color:{'#27AE60' if safety_now=='SAFE' else '#E67E22' if safety_now=='CAUTION' else '#C0392B'};font-size:0.8rem;font-weight:700">
    SAFETY: {safety_now}
  </div>
</div>
<div style="background:#101B26;border:1px solid #1C3D52;border-radius:6px;padding:8px;
            font-family:'IBM Plex Mono',monospace;font-size:0.68rem;line-height:1.7;margin-top:4px">
  <div>Wall dist: <span style="color:{'#27AE60' if tel.get('tof',0)>120 else '#E67E22' if tel.get('tof',0)>60 else '#C0392B'}">{tel.get('tof',0)}cm</span></div>
  <div>Alt: {tel.get('height',0)}cm (limit {st.session_state['safety_max_alt']}cm)</div>
  <div>Battery: {bat:.0f}% (land@{st.session_state['min_battery_rtl']}%)</div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">🔔 ALERTS</div>', unsafe_allow_html=True)
        for a in st.session_state["alerts"][:10]:
            css = {"crit": "alert-crit", "warn": "alert-warn", "ok": "alert-ok"}.get(a["level"], "alert-info")
            st.markdown(f'<div class="{css}">[{a["ts"]}] {a["msg"]}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB — TELEMETRY  (every sensor Tello reports)
# ════════════════════════════════════════════════════════════════════════════
with tab_telem:
    t1, t2 = st.columns(2, gap="medium")

    with t1:
        st.markdown('<div class="sec-hdr">📡 FLIGHT SENSORS</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:8px">
  {kpi(f"{tel.get('battery',0)}%","Battery",color="#27AE60")}
  {kpi(f"{tel.get('height',0)} cm","Altitude",color="#2E86AB")}
  {kpi(f"{tel.get('tof',0)} cm","Wall/ToF Dist",color="#E67E22")}
  {kpi(f"{tel.get('yaw',0):.0f}°","Yaw",color="#2E86AB")}
  {kpi(f"{tel.get('pitch',0):.0f}°","Pitch",color="#D5DEE6")}
  {kpi(f"{tel.get('roll',0):.0f}°","Roll",color="#D5DEE6")}
  {kpi(f"{tel.get('speed_x',0)} cm/s","Speed X",color="#2E86AB")}
  {kpi(f"{tel.get('speed_y',0)} cm/s","Speed Y",color="#2E86AB")}
  {kpi(f"{tel.get('speed_z',0)} cm/s","Speed Z",color="#2E86AB")}
  {kpi(f"{tel.get('accel_x',0):.1f}","Accel X",color="#D5DEE6")}
  {kpi(f"{tel.get('accel_y',0):.1f}","Accel Y",color="#D5DEE6")}
  {kpi(f"{tel.get('accel_z',0):.1f}","Accel Z",color="#D5DEE6")}
</div>
""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flight Time", f"{tel.get('flight_time',0)}s")
        c2.metric("Distance (m)", f"{stats['flight_distance_m']:.1f}")
        c3.metric("Temp", f"{tel.get('temp_lo',0)}-{tel.get('temp_hi',0)}°C")
        c4.metric("Barometer", f"{tel.get('baro',0):.1f}")
        st.metric("WiFi SNR", str(tel.get("wifi_snr", "—")))

    with t2:
        st.markdown('<div class="sec-hdr">📈 ALTITUDE / BATTERY HISTORY</div>', unsafe_allow_html=True)
        fl = st.session_state.get("flight_log", [])
        if fl:
            df_fl = pd.DataFrame(fl[-300:])
            if PLOTLY_AVAILABLE and "height" in df_fl.columns and "battery" in df_fl.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df_fl["height"], name="Altitude (cm)", line=dict(color="#2E86AB")))
                fig.add_trace(go.Scatter(y=df_fl["battery"], name="Battery %", line=dict(color="#27AE60"), yaxis="y2"))
                fig.update_layout(height=260, paper_bgcolor="#0B1420", plot_bgcolor="#101B26",
                                   font=dict(color="#D5DEE6", size=10),
                                   xaxis=dict(gridcolor="#1C3D52", showgrid=True),
                                   yaxis=dict(gridcolor="#1C3D52"), yaxis2=dict(overlaying="y", side="right"),
                                   margin=dict(l=30, r=30, t=20, b=20), legend=dict(bgcolor="#101B26"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df_fl.set_index("time")[["height", "battery"]] if "time" in df_fl.columns else df_fl)
        else:
            st.info("Flight log chart will appear once flying.")

        st.markdown('<div class="sec-hdr">⬇️ EXPORT</div>', unsafe_allow_html=True)
        if fl:
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=fl[0].keys())
            w.writeheader(); w.writerows(fl)
            st.download_button("⬇️ Download Flight Log CSV", data=buf.getvalue(),
                               file_name=f"flight_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv", key="dl_flight_csv", use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#1C3D52;margin:16px 0 6px'>", unsafe_allow_html=True)
st.caption(
    f"🚁 Tello Auto Drive · djitellopy {'✅' if TELLO_AVAILABLE else '❌ not installed'} · "
    f"OpenCV {'✅' if CV2_AVAILABLE else '❌ not installed'} · "
    f"Session: {st.session_state.get('reconnect_count',0)} connect(s) · "
    f"Draw-your-own-path Auto Drive · Live Camera · Wall-distance holding · Full sensor telemetry"
)
