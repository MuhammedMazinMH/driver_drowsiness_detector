
# Driver Drowsiness Detector (EAR + Yawn CNN)

This lightweight Python project detects driver drowsiness in real-time by combining two cues:

1. **Eye Aspect Ratio (EAR)** – classical geometry to flag prolonged eye-closure.
2. **Tiny CNN Yawn Classifier** – TFLite model that classifies mouth crops as *yawning / not-yawning*.

The system runs at **≥30 FPS** on laptop CPUs and raises an audible alarm when either cue indicates drowsiness.

---
## Quick-Start

```bash
# 1. Unzip this folder
cd drowsiness_detector_project

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the detector (web-cam index 0 by default)
cd src
python main.py 
```

If this is the **first** run, the script automatically downloads a pre-trained `yawn_detector.tflite` (~380 kB) into `models/`. Internet connectivity is required for that one-time step.

---
## Files & Folders
| Path | Purpose |
|------|---------|
| `requirements.txt` | Python package pins |
| `src/main.py` | End-to-end inference pipeline |
| `src/ear.py` | EAR computation helper |
| `src/utils.py` | Misc. helpers (landmark indices, download) |
| `models/` | Holds the TFLite CNN model |
| `assets/alarm.wav` | Sample beep used for alert |

---
## Arguments (main.py)
```
--camera      Video-capture index (default 0)
--ear_thresh  EAR threshold (default 0.20)
--ear_frames  Frames EAR < thresh before counting an eye event (default 48)
--yawn_frames Consecutive yawning frames to trigger yawn event (default 10)
--cooldown    Seconds after alarm before re-arming (default 60)
```

---
## Tested Environment
* Python 3.11
* Ubuntu 22.04 & Windows 11
* Intel® Core™ i5-8250U (15 W TDP) – **31 FPS**

---
## Troubleshooting
1. **`No module named mediapipe`** – ensure installation succeeded; consider `pip install mediapipe==0.12.0`.
2. **Low FPS (<15)** – reduce resolution inside `main.py` (set 480x360).
3. **Playsound on Linux blocks** – change to `aplay` or `ffplay` in `utils.play_alert()`.

---
MIT License © 2025 Minor-Project-B Team
