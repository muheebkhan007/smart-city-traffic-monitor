"""
app.py
-------
Smart City Traffic Monitor — Streamlit Dashboard
CPU-optimised | SORT Tracking | Line Crossing Counter | CSV Logging

Run:
    streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime

from modules.detector import VehicleDetector
from modules.tracker  import SORTTracker
from modules.counter  import LineCounter
from modules.utils    import (FPSCounter, annotate_frame,
                               init_csv, log_event, read_csv, clear_csv, LOG_FILE)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Traffic Monitor",
    page_icon="🚦",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background-color: #0f1117; }
.block-container { padding-top: 0.8rem; }
.metric-box {
    background: #1c1f26; border: 1px solid #2d3139;
    border-radius: 10px; padding: 1rem; text-align: center;
}
.metric-box h1 { color: #4fc3f7; font-size: 2.4rem; margin: 0; }
.metric-box p  { color: #9ea3b0; margin: 0; font-size: 0.82rem; }
.status-ok  { color: #66bb6a; font-weight: bold; }
.status-off { color: #ef5350; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Init CSV
init_csv()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # Video source
    source_opt = st.radio("📹 Video Source", ["Upload Video", "Webcam (0)"])
    uploaded   = None
    if source_opt == "Upload Video":
        uploaded = st.file_uploader("Choose file",
                                    type=["mp4","avi","mov","mkv"])

    st.divider()
    st.markdown("### 🎯 Detection")
    conf_thresh  = st.slider("Confidence",   0.15, 0.80, 0.30, 0.05)
    skip_frames  = st.selectbox("Process Every N Frames", [1, 2, 3], index=1)
    line_pos     = st.slider("Counting Line Position", 0.30, 0.80, 0.55, 0.05,
                              help="Fraction of frame height")

    st.divider()
    st.markdown("### 🔍 SORT Tracker")
    max_age      = st.slider("Max Age (frames)", 5, 30, 10)
    iou_thresh   = st.slider("IoU Threshold",    0.10, 0.50, 0.25, 0.05)

    st.divider()
    col_r, col_c = st.columns(2)
    reset_btn    = col_r.button("🔄 Reset",     use_container_width=True)
    clear_btn    = col_c.button("🗑️ Clear CSV", use_container_width=True)

    if clear_btn:
        clear_csv()
        st.success("CSV cleared!")

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if reset_btn:
    st.session_state.running = False
    st.session_state["reset_flag"] = True

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🚦 Smart City Traffic Monitor")
st.caption("CPU-Optimised · SORT Tracking · Line Crossing Counter · CSV Logging")
st.divider()

tab_live, tab_logs, tab_analytics = st.tabs(
    ["📹 Live Detection", "📋 Event Logs", "📊 Analytics"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:

    # ── Metric cards ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    ph_car   = m1.empty()
    ph_moto  = m2.empty()
    ph_bus   = m3.empty()
    ph_truck = m4.empty()
    ph_total = m5.empty()

    def render_metrics(counts):
        ph_car.markdown(
            f'<div class="metric-box"><h1>🚗 {counts.get("car",0)}</h1><p>Cars</p></div>',
            unsafe_allow_html=True)
        ph_moto.markdown(
            f'<div class="metric-box"><h1>🏍️ {counts.get("motorcycle",0)}</h1><p>Motorcycles</p></div>',
            unsafe_allow_html=True)
        ph_bus.markdown(
            f'<div class="metric-box"><h1>🚌 {counts.get("bus",0)}</h1><p>Buses</p></div>',
            unsafe_allow_html=True)
        ph_truck.markdown(
            f'<div class="metric-box"><h1>🚛 {counts.get("truck",0)}</h1><p>Trucks</p></div>',
            unsafe_allow_html=True)
        ph_total.markdown(
            f'<div class="metric-box"><h1>🚦 {counts.get("total",0)}</h1><p>Total</p></div>',
            unsafe_allow_html=True)

    render_metrics({})
    st.divider()

    # ── Video + controls ──────────────────────────────────────────────────────
    vid_col, ctrl_col = st.columns([3, 1])

    with vid_col:
        frame_ph  = st.empty()
        status_ph = st.empty()

    with ctrl_col:
        st.markdown("### Controls")
        start_btn = st.button("▶️ Start", use_container_width=True, type="primary")
        stop_btn  = st.button("⏹️ Stop",  use_container_width=True)
        st.divider()
        st.markdown("### Recent Events")
        events_ph = st.empty()

    if start_btn: st.session_state.running = True
    if stop_btn:  st.session_state.running = False

    # ── Main detection loop ───────────────────────────────────────────────────
    if st.session_state.running:
        # Resolve video source
        if source_opt == "Webcam (0)":
            src = 0
        elif uploaded:
            tmp = Path("output") / uploaded.name
            tmp.parent.mkdir(exist_ok=True)
            tmp.write_bytes(uploaded.read())
            src = str(tmp)
        else:
            st.warning("⚠️ Please upload a video file first.")
            st.session_state.running = False
            st.stop()

        # Init modules
        try:
            detector = VehicleDetector("yolov8n.pt", conf=conf_thresh)
        except Exception as e:
            st.error(f"Model load error: {e}")
            st.stop()

        tracker  = SORTTracker(max_age=max_age, iou_thresh=iou_thresh)
        fps_calc = FPSCounter()
        cap      = cv2.VideoCapture(src)

        if not cap.isOpened():
            st.error("Cannot open video source!")
            st.stop()

        frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y   = int(frame_h * line_pos)
        counter  = LineCounter(line_y=line_y, frame_w=frame_w)

        frame_idx    = 0
        recent_evs   = []

        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    status_ph.info("✅ Video finished.")
                    break

                frame_idx += 1
                if frame_idx % skip_frames != 0:
                    continue

                # ── Pipeline ──────────────────────────────────────────────
                dets    = detector.detect(frame)
                tracks  = tracker.update(dets)
                events  = counter.update(tracks)

                # Log crossing events to CSV
                for ev in events:
                    log_event(ev, counter.total)
                    recent_evs.insert(0, f"#{ev['track_id']} {ev['label'].upper()}")

                fps = fps_calc.tick()

                # ── Annotate ──────────────────────────────────────────────
                frame = detector.draw(frame, tracks)
                frame = counter.draw(frame)
                frame = annotate_frame(frame, fps)

                # ── Display ───────────────────────────────────────────────
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_ph.image(rgb, use_container_width=True)

                # Update metrics
                counts_with_total = {**counter.counts, "total": counter.total}
                render_metrics(counts_with_total)

                status_ph.markdown(
                    f'<span class="status-ok">● LIVE</span>  '
                    f'FPS: **{fps:.1f}** | '
                    f'Tracked: **{len(tracks)}** | '
                    f'Total crossed: **{counter.total}**',
                    unsafe_allow_html=True
                )

                events_ph.markdown(
                    "\n".join(f"`{e}`" for e in recent_evs[:8]) or "_None yet_"
                )

        finally:
            cap.release()
            status_ph.markdown('<span class="status-off">● STOPPED</span>',
                               unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LOGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.subheader("📋 Vehicle Crossing Event Log")

    col_ref, col_dl = st.columns([1, 1])
    if col_ref.button("🔄 Refresh Logs"):
        st.rerun()

    rows = read_csv()
    if not rows:
        st.info("No events logged yet. Start detection to populate logs.")
    else:
        df = pd.DataFrame(rows)
        df.columns = ["Timestamp", "Vehicle Type", "Track ID", "Confidence", "Total Count"]

        # Summary
        summary = df["Vehicle Type"].value_counts().to_dict()
        c1,c2,c3,c4,c5 = st.columns(5)
        for col, (vt, icon) in zip(
            [c1,c2,c3,c4],
            [("car","🚗"),("motorcycle","🏍️"),("bus","🚌"),("truck","🚛")]
        ):
            col.markdown(
                f'<div class="metric-box"><h1>{icon} {summary.get(vt,0)}</h1>'
                f'<p>{vt.capitalize()}s</p></div>',
                unsafe_allow_html=True)
        c5.markdown(
            f'<div class="metric-box"><h1>🚦 {len(df)}</h1><p>Total Events</p></div>',
            unsafe_allow_html=True)

        st.divider()
        st.dataframe(df, use_container_width=True, height=380)

        col_dl.download_button(
            "⬇️ Download CSV",
            data=open(LOG_FILE, "rb").read(),
            file_name=f"traffic_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.subheader("📊 Traffic Analytics")

    rows = read_csv()
    if not rows:
        st.info("No data yet.")
    else:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"]      = df["timestamp"].dt.floor("h")
        df["confidence"] = df["confidence"].astype(float)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("##### Vehicle Type Distribution")
            vc = df["vehicle_type"].value_counts().reset_index()
            vc.columns = ["Type", "Count"]
            st.bar_chart(vc.set_index("Type"))

        with c2:
            st.markdown("##### Hourly Traffic Volume")
            hourly = df.groupby("hour").size().reset_index(name="Count")
            st.line_chart(hourly.set_index("hour"))

        st.divider()
        st.markdown("##### Hourly Breakdown by Vehicle Type")
        stacked = (df.groupby(["hour","vehicle_type"])
                     .size().unstack(fill_value=0))
        st.area_chart(stacked)

        st.divider()
        a, b, c_ = st.columns(3)
        a.metric("Total Events",   len(df))
        b.metric("Unique Vehicles", df["track_id"].nunique())
        c_.metric("Avg Confidence", f"{df['confidence'].mean():.0%}")
