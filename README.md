# DJI Tello · Auto Drive

Single-file Streamlit app for flying a DJI Tello, with a drawn-path
**Auto Drive** mode. Rebuilt from two source apps — YOLO defect
detection, analytics, reports, weather, and the AI chat assistant were
removed. "AI Autopilot" is renamed **Auto Drive** throughout.

## What it has

- **Connection & full sensor telemetry** — battery, height, ToF
  distance, barometer, flight time, temperature, pitch/roll/yaw,
  velocity (x/y/z), acceleration (x/y/z), and mission-pad data on
  Tello EDU units — pulled in one call via `get_current_state()`.
- **Manual flight control** — takeoff/land, translate in all 6
  directions, rotate, flip, emergency stop.
- **Live Feed tab** — video stream, snapshot, and video recording.
- **Auto Drive tab** — draw a path on a grid (click cells in flight
  order), preview it, then fly *exactly* that path — the drone does
  not deviate onto any other route. Video records automatically for
  the whole run. Before every leg it checks the ToF sensor and a
  configurable wall-safety margin baked into the grid, and aborts the
  mission (lands safely) if either is violated.
- **Mission Log tab** — plain event log of connects, moves, waypoints,
  and safety aborts.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Field deployment steps

1. Power on the Tello and wait for the LED to flash.
2. On the laptop/tablet that will run this app, join the Tello's own
   Wi-Fi network (`TELLO-XXXXXX`). The Tello has no internet uplink,
   so this app must run on a machine joined to the drone's network —
   it cannot be hosted on a remote server reached over the internet.
3. Start the app (`streamlit run app.py`) and open it in a browser on
   that same machine.
4. Click **Connect** in the sidebar.
5. Fly manually from the sidebar, or open **Auto Drive**, draw a path,
   set the wall margin and cell size to match the real room, and press
   **Start Auto Drive**.
6. Recordings save to `recordings/`, snapshots to `snapshots/`, both
   created next to `app.py` on first run.

## Note on wall clearance

The stock Tello's only distance sensor (ToF) faces downward, not
forward — there is no built-in forward obstacle sensor on this
airframe. "Keeping distance from the wall" is therefore enforced two
ways here: (1) the grid margin you set on the drawn path, which keeps
the *planned* route away from the room's edges, and (2) a live ToF
check before each leg as a secondary safety net. For hard,
sensor-verified forward-obstacle avoidance you'd need a Tello variant
or add-on with a forward-facing rangefinder.
