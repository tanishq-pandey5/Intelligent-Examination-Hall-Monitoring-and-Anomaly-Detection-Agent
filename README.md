# 🎓 Intelligent Examination Hall Monitoring and Anomaly Detection Agent

**AI Examination Video Frame Analysis and Pattern Detection System**

An AI-assisted, model-based examination hall monitoring and video frame analysis application that analyzes CCTV video footage to detect people, track candidate movements, identify potentially unusual examination-related behavior patterns across time, and generate detailed analysis reports with timestamps and forensic evidence frames.

> **Important Notice**: This system is an examination-support prototype, not an automated cheating verdict system. Every alert remains reviewable by an authorized human invigilator.

---

## 📋 Table of Contents
- [Project Overview & Objectives](#project-overview--objectives)
- [Technology Stack](#technology-stack)
- [Project Architecture & Structure](#project-architecture--structure)
- [Behavioral Analysis Modules (Section 4)](#behavioral-analysis-modules-section-4)
- [Temporal Pattern Analysis & Model-Based Agent](#temporal-pattern-analysis--model-based-agent)
- [Production Rules & Reasoning Engine](#production-rules--reasoning-engine)
- [Interpretable Risk Scoring Formula](#interpretable-risk-scoring-formula)
- [Explainable Output Format](#explainable-output-format)
- [Installation & Quickstart](#installation--quickstart)
- [Dashboard Walkthrough (Section 10)](#dashboard-walkthrough-section-10)
- [CLI Processing & Benchmark Simulation](#cli-processing--benchmark-simulation)
- [Test Suite & System Evaluation (Section 12)](#test-suite--system-evaluation-section-12)
- [Database Schema & Reporting](#database-schema--reporting)

---

## 🎯 Project Overview & Objectives

Traditional CCTV review relies on proctors manually scanning dozens of video streams for hours. This application assists proctors by:
1. Ingesting examination-hall CCTV videos (MP4, AVI, MOV) without loading entire files into memory.
2. Detecting people and mobile phones on sampled frames using pretrained **Ultralytics YOLOv8**.
3. Tracking candidates across consecutive frames using **ByteTrack** with temporary tracking IDs.
4. Estimating candidate ground **foot points** and verifying adherence to configured seat regions.
5. Detecting 5 target examination behaviors:
   - **Leaving the Assigned Seat** (foot point departed for $>5\,\text{s}$).
   - **Repeated Head Turning** (head-pose estimation and distinct side glances).
   - **Mobile Phone Presence vs Confirmed Use** (spatial-temporal workspace association).
   - **Multiple Students Gathering** (unusual physical proximity clustering).
   - **Unusual Movement** (abrupt displacement and velocity spikes).
6. Evaluating temporal patterns over sliding windows rather than flagging single isolated frames.
7. Maintaining a **Model-Based Agent** per candidate with a 4-tier state machine:
   $$\text{NORMAL} \longrightarrow \text{OBSERVE} \longrightarrow \text{SUSPICIOUS} \longrightarrow \text{ALERT}$$
8. Generating human-readable explainable alerts, saving high-resolution evidence snapshots, and exporting CSV reports.

---

## 🛠️ Technology Stack

- **Programming Language**: Python 3.10+
- **Video & Frame Processing**: OpenCV (`cv2`) with generator-based frame sampling
- **Object Detection**: Ultralytics YOLOv8 (pretrained `yolov8n.pt` for `person` and `cell phone` classes)
- **Multi-Object Tracking**: ByteTrack / IoU centroid matching with trajectory velocity calculation
- **Head Pose Estimation**: Lateral facial centroid displacement & Sobel edge gradient asymmetry
- **Invigilator Dashboard**: Streamlit interactive web interface
- **Feature Extraction & Analysis**: NumPy, Pandas, SciPy, Scikit-learn
- **Data Persistence**: SQLite (`data/exam_monitor.db`)
- **Visualizations**: Plotly Express & KaTeX

---

## 📁 Project Architecture & Structure

```
├── app.py                      # Streamlit video upload and analysis dashboard
├── config/
│   ├── __init__.py
│   └── settings.py             # Centralized thresholds, weights, and production rules
├── src/
│   ├── __init__.py
│   ├── video_input.py          # Video format validation, resolution, FPS, metadata
│   ├── frame_sampler.py        # Memory-safe frame sampling at configurable FPS & timecodes
│   ├── detector.py             # Ultralytics YOLOv8 person & mobile-phone detector
│   ├── tracker.py              # ByteTrack multi-person tracker with foot points & history
│   ├── seat_mapper.py          # Manual/grid seat regions, foot-point projection & occupancy
│   ├── behavior.py             # 5 behavioral modules (Seat departure, Head turn, Phone, Gathering, Motion)
│   ├── agent.py                # Model-Based Agent with FSM (NORMAL->OBSERVE->SUSPICIOUS->ALERT)
│   ├── reasoning.py            # Rule-based reasoning engine with configurable IF-THEN rules
│   ├── risk_engine.py          # Interpretable 0-100 risk scoring with factor breakdown
│   ├── explanations.py         # Human-readable explainable alerts (Section 9 format)
│   ├── evidence.py             # Forensic evidence snapshots, deduplication & retention
│   ├── database.py             # SQLite persistence for sessions, events, and reviews
│   └── report_generator.py     # Downloadable CSV report and Markdown session summary
├── tests/
│   ├── test_video_input.py     # Tests for video validation and frame sampling
│   ├── test_detector_tracker.py# Tests for YOLO detection and ByteTrack tracking
│   ├── test_behavior.py        # Tests for 5 behavioral modules and seat mapping
│   ├── test_agent.py           # Tests for FSM transitions and temporal decay
│   ├── test_reasoning_risk.py  # Tests for production rules and risk scoring
│   └── test_database_report.py # Tests for SQLite operations and CSV reports
├── data/
│   ├── evidence/               # Saved evidence frame snapshots (.jpg)
│   ├── sample_videos/          # Benchmark simulation video files (.mp4)
│   └── exam_monitor.db         # Persistent SQLite database
├── outputs/
│   └── reports/                # Generated CSV reports and session summaries
├── requirements.txt            # Python dependencies
└── README.md                   # Complete system documentation
```

---

## 🔍 Behavioral Analysis Modules (Section 4)

### A. Leaving the Assigned Seat
- **Foot-Point Estimation**: Calculates ground foot coordinate $(x_{\text{center}}, y_{\text{bottom}})$.
- **Boundary Verification**: Evaluates whether foot point lies outside the configured seat polygon.
- **Duration & Return Time**: Tracks exact departure duration and records return timestamp when the candidate re-enters the seat.
- **Grace Period**: Ignores brief boundary glances ($<2.0\,\text{s}$).

### B. Repeated Head Turning
- **Head-Pose Estimation**: Analyzes lateral facial asymmetry relative to the seat center axis.
- **Oscillatory Filtering**: Distinguishes distinct turn episodes (e.g. glancing at peer, looking back at paper, glancing again).
- **Windowed Recurrence**: Accumulates turn counts over a $25\,\text{s}$ sliding window.
- **Limitation Reporting**: Flags when camera angle or resolution restricts reliable pose estimation.

### C. Mobile Phone Detection vs Confirmed Use
- **Workspace Association**: Associates detected phones with a candidate only if located within their desk/workspace bounding area.
- **Temporal Persistence**: Requires multi-frame persistence ($\ge 4$ frames or $\ge 1.5\,\text{s}$) to eliminate single-frame false positives.
- **Use Categorization**: Clearly distinguishes between suspected presence (phone on desk) and confirmed active use (sustained interaction $>4\,\text{s}$).

### D. Multiple Students Gathering
- **Pairwise Distance Matrix**: Calculates Euclidean distances between all candidate centroids.
- **Cluster Identification**: Triggers when $\ge 2$ students remain within $100\,\text{px}$ proximity outside normal desk separation.
- **Duration Tracking**: Measures cluster persistence.

### E. Unusual Movement
- **Kinematic Analysis**: Monitors displacement vectors and velocity spikes ($>40\,\text{px/s}$).
- **Erratic Motion Damping**: Ignores momentary normal twitches while capturing sustained agitation, stretching across desks, or passing notes.

---

## 🧠 Temporal Pattern Analysis & Model-Based Agent

### State Machine Lifecycle
Each candidate's state evolves according to accumulated observations:

```
      Score < 25               25 <= Score < 55            55 <= Score < 80            Score >= 80
+---------------------+     +--------------------+     +---------------------+     +-------------------+
|       NORMAL        | --> |      OBSERVE       | --> |      SUSPICIOUS     | --> |       ALERT       |
| (Green #2ecc71)     | <-- | (Amber #f1c40f)    | <-- | (Orange #e67e22)    | <-- | (Crimson #e74c3c) |
+---------------------+     +--------------------+     +---------------------+     +-------------------+
       ^                            |                           |                            |
       +----------------------------+---------------------------+----------------------------+
                                   Risk Decay During Calm Behavior (-2.5 pts/s)
```

### Episodic Grace Window Damping
Repeated cheating behavior often involves intermittent pauses (e.g., student turns to neighbor, glances at paper, turns again). The agent uses a **2.5-second episodic grace window** so duration does not reset to zero during brief glances back.

### Human Review States
Every logged event maintains an actionable review status:
- `UNREVIEWED` $\to$ `ACKNOWLEDGED` $\to$ `UNDER_REVIEW` $\to$ `RESOLVED` / `FALSE_POSITIVE`

---

## 📜 Production Rules & Reasoning Engine

Configured in [`config/settings.py`](file:///Users/tanishqpandey/Documents/Projects/Intelligent%20Examination%20Hall%20Monitoring%20and%20Anomaly%20Detection%20Agent/config/settings.py):

- **Rule 1 (Seat Departure)**: $\mathbf{IF} \; \text{duration} \ge 5.0\,\text{s} \; \mathbf{THEN} \; \text{Event} = \text{LEAVING\_SEAT}$.
- **Rule 2 (Repeated Turning)**: $\mathbf{IF} \; \text{turn\_count} \ge 3 \; \mathbf{AND} \; \text{duration} \ge 4.0\,\text{s} \; \mathbf{THEN} \; \text{State} = \text{SUSPICIOUS/ALERT}$.
- **Rule 3 (Mobile Phone)**: $\mathbf{IF} \; \text{phone\_detected} \; \mathbf{AND} \; \text{frames} \ge 4 \; \mathbf{THEN} \; \text{State} = \text{ALERT}$.
- **Rule 4 (Group Gathering)**: $\mathbf{IF} \; \text{cluster\_size} \ge 2 \; \mathbf{AND} \; \text{duration} \ge 3.5\,\text{s} \; \mathbf{THEN} \; \text{Event} = \text{GATHERING\_CLUSTER}$.
- **Rule 5 (Unusual Movement)**: $\mathbf{IF} \; \text{velocity} \ge 40\,\text{px/s} \; \mathbf{AND} \; \text{duration} \ge 3.0\,\text{s} \; \mathbf{THEN} \; \text{State} = \text{SUSPICIOUS}$.
- **Rule 6 (Composite Escalation)**: $\mathbf{IF} \; \text{active\_anomalies} \ge 2 \; \mathbf{THEN} \; \text{State} = \text{ALERT}$ (Escalated Priority).

---

## 🧮 Interpretable Risk Scoring Formula

The risk score ($0-100$) provides full mathematical transparency:

$$\text{Base Score} = w_c \cdot C_{\text{det}} + w_d \cdot \min\left(\frac{D}{D_{\text{target}}}, 1.0\right) + w_f \cdot \min\left(\frac{F}{F_{\text{target}}}, 1.0\right) + w_t \cdot S_{\text{type}} + w_r \cdot R_{\text{track}}$$

where:
- $C_{\text{det}}$: Detection confidence ($w_c = 0.25$)
- $D$: Sustained episodic duration ($w_d = 0.30$)
- $F$: Recurrence count in sliding window ($w_f = 0.20$)
- $S_{\text{type}}$: Activity severity multiplier ($w_t = 0.15$)
- $R_{\text{track}}$: Tracking reliability metric ($w_r = 0.10$)

$$\text{Final Risk Score} = \min\left(100.0, \; \text{Base Score} \times (0.85 + 0.15 \times \text{Consistency}) \times 100.0\right)$$

---

## 📋 Explainable Output Format (Section 9)

Every logged anomaly produces an explainable structured alert:

```text
Event: Repeated side movement
Seat: 18
Timestamp: 00:12:36
Duration: 14 seconds
Detection confidence: 86%
Risk score: 68/100
Reason: Repeated side movement was observed during the configured analysis window. Human verification is recommended.
Evidence: data/evidence/seat_18_001236.jpg
```

---

## 🚀 Installation & Quickstart

```bash
# 1. Clone repository and navigate to directory
cd "/Users/tanishqpandey/Documents/Projects/Intelligent Examination Hall Monitoring and Anomaly Detection Agent"

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Train/Fine-Tune Models on Classroom Videos
python3 train.py --videos videos/ --epochs 5 --mode all

# 4. Launch the Streamlit Invigilator Dashboard
python3 -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧠 Model Training via Videos Folder (`train.py`)

The system supports end-to-end training and domain adaptation directly on classroom recordings located in `videos/`:

```bash
# Full pipeline: fine-tunes YOLOv8 and fits behavioral anomaly model
python3 train.py --videos videos/ --epochs 5 --mode all

# Train unsupervised ML behavioral anomaly detector only
python3 train.py --mode behavior

# Fine-tune YOLO object detector only
python3 train.py --mode yolo --epochs 5
```

### Trained Artifacts Generated:
1. `models/yolov8n_exam.pt`: Custom fine-tuned YOLOv8 weights adapted to university classroom lighting and student desk arrangements.
2. `models/behavior_anomaly_detector.joblib`: 10-dimensional `IsolationForest` + `MinCovDet` statistical anomaly detector trained on student velocity, trajectory jitter, proximity to neighbors, and displacement.
3. `models/training_meta.json`: Full training run metadata, loss curves, and feature distribution boundaries.

---

## 🖥️ Dashboard Walkthrough (Section 10)

1. **📹 Video Stream & Frame Analysis**:
   - Select videos directly from the `videos/` folder, load the 18-desk benchmark video, or upload `.mp4`, `.avi`, `.mov`.
   - Toggle between **Fine-Tuned Exam YOLOv8** and **Base YOLOv8**, or enable the **ML Behavioral Anomaly Detector**.
   - Annotated frames display tracking IDs, bounding boxes, foot-point coordinates, and seat zone outlines.
   - Interactive timeline scrubber and **"Jump to Anomaly Event"** selector.
2. **🗺️ Examination Hall Seating Layout**:
   - 2D floor plan view of the hall (18 desks).
   - Seats dynamically highlight and pulse in red/orange when suspicious behavior occurs.
   - Interactive Candidate Dossier inspecting individual risk trajectories.
3. **🚨 Explainable Anomaly Events Feed**:
   - Filter events by Seat, Risk Category, Event Type, or Review Status.
   - Invigilator action buttons: **Acknowledge**, **Under Review**, **False Positive**, **Mark Resolved**.
4. **📸 Evidence Snapshot Vault**:
   - Gallery of captured forensic evidence frames with targeting graphics.
5. **📊 Session Summary & CSV Reports**:
   - Download the official incident audit log as a **CSV file**.
   - Download the human-readable session summary in **Markdown**.
6. **🧠 Model Training & Video Calibration**:
   - Video inventory of `videos/class_1.mp4` through `class_7.mp4`.
   - One-click model retraining with configurable epochs and frame sampling intervals.
   - Inspection of learned normality baselines (means, standard deviations, and 99th percentile anomaly thresholds).


---

## 💻 CLI Processing & Benchmark Simulation

### Generate Benchmark 18-Desk Examination Hall Video:
```bash
python3 run_demo.py --generate-video --duration 20.0
```

### Process Video Offline from Terminal:
```bash
python3 run_demo.py --input data/sample_videos/exam_hall_benchmark.mp4 --sampling-fps 5.0
```

---

## 🧪 Test Suite & System Evaluation (Section 12)

Run all 30 automated unit tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

Output:
```text
Ran 30 tests in 3.787s

OK
```

### Verified Test Modules:
- `tests/test_video_input.py`: Format validation, resolution/FPS metadata extraction, frame sampler intervals, and timecode calculation.
- `tests/test_detector_tracker.py`: YOLO person/phone detection, temporary tracking ID assignment, foot-point projection, trajectory history, and occlusion handling.
- `tests/test_behavior.py`: 5 behavioral modules (Seat departure duration, head-turn episodes, phone workspace association, gathering clusters, unusual motion).
- `tests/test_agent.py`: Model-Based Agent FSM state transitions, single-frame damping, and risk decay.
- `tests/test_reasoning_risk.py`: Production rule evaluation, multi-factor risk scoring, non-verdict disclaimer, and Section 9 output format verification.
- `tests/test_database_report.py`: SQLite session and event CRUD, review status updates, CSV export, and evidence deduplication.

---

## 🗄️ Database Schema & Reporting

### `video_sessions` Table
Stores metadata for each processed video: `session_id`, `file_name`, `fps`, `resolution`, `duration_seconds`, `total_frames`, `analyzed_frames`, `sampling_fps`.

### `analysis_events` Table
Stores each flagged incident: `id`, `session_id`, `video_timestamp_sec`, `timecode_str`, `tracking_id`, `seat_id`, `event_type`, `event_display_name`, `confidence`, `duration_seconds`, `risk_score`, `risk_category`, `reason`, `evidence_path`, `review_status`, `invigilator_notes`.

---

## 👥 Academic & Research Disclaimer
Developed for University Examination Integrity and AI Classroom Monitoring Research. Licensed under the MIT License.
