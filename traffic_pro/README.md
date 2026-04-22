# 🚦 Smart City Traffic Monitor (CPU-Optimised)

Real-time vehicle detection, SORT tracking, and line-crossing counter.
Runs fully on CPU — no GPU required.

---

## 📁 Project Structure

```
traffic_pro/
│
├── app.py                  ← Run this (Streamlit UI)
│
├── modules/
│   ├── detector.py         ← YOLOv8 vehicle detection (CPU optimised)
│   ├── tracker.py          ← SORT tracker (Kalman + Hungarian)
│   ├── counter.py          ← Virtual line crossing counter
│   └── utils.py            ← CSV logging + FPS + frame annotation
│
├── logs/
│   └── vehicle_log.csv     ← Auto-generated event log
│
├── data/                   ← Put traffic videos here
├── output/                 ← Temp video uploads
└── requirements.txt
```

---

## ⚙️ Setup (Python 3.11)

```bash
# 1. Create venv
py -3.11 -m venv venv
venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

---

## 🚀 Features

| Feature | Detail |
|---|---|
| Detection | YOLOv8n — car, motorcycle, bus, truck |
| Tracking | SORT (Kalman Filter + Hungarian Algorithm) |
| Counting | Virtual horizontal line — each vehicle counted once |
| Logging | CSV file with timestamp, type, track ID |
| Dashboard | Live video, metrics, logs, analytics |
| CPU Speed | ~8-15 FPS on average laptop |

---

## 🎛️ Sidebar Controls

- **Confidence** — lower = detect more, higher = more accurate
- **Skip Frames** — process every 2nd/3rd frame for speed
- **Line Position** — where counting line is drawn
- **Max Age** — how long to keep a lost track alive
- **Reset** — reset counters
- **Clear CSV** — wipe the log file
