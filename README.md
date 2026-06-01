# Universal Relationship Graph Intelligence System (URG-IS)

> "Intelligence is not about capturing the past; it is about automating the anticipation of the future." — **URG-IS Deployed Core Principle**

URG-IS is a state-of-the-art, edge-accelerated, **Multi-Agent Predictive Surveillance Platform** designed to operate entirely on-premises. Built to comply with strict data privacy guidelines, including India's **Digital Personal Data Protection (DPDP) Act**, it processes real-time surveillance video streams completely locally—ensuring that sensitive tracking coordinates and telemetry data never leave your localized physical architecture.

---

## 🛠️ Complete Technical Stack

| Pipeline Layer | Technology / Model Implemented | System Function |
| :--- | :--- | :--- |
| **Ingestion Engine** | `OpenCV` & `FFmpeg` | Captures live camera RTSP streams, handling sequential decoding. |
| **Object Detection** | `Ultralytics YOLOv8` (`yolov8s.pt`) | High-speed, frame-level localized scanning to draw entity bounding boxes. |
| **Intra-Cam Tracking** | `BoT-SORT` | Persists unique pedestrian tracking IDs across frame transformations. |
| **Global Re-ID** | `OSNet` (TorchReID) + `FAISS` | Extracts a 512-D visual fingerprint to re-identify entities across cameras. |
| **Sovereign Database** | `NetworkX` | High-speed, in-memory graph repository managing relation community clusters via **Louvain Partitioning**. |
| **Predictive Foundation**| `amazon/chronos-bolt-tiny` | Loaded locally to execute zero-shot multi-step time-series trajectory forecasting. |
| **Cognition Layer** | `Ollama` (`Llama 3.2 3B Instruct`) | Translates mathematical graph relationships into plain-English operational briefs. |
| **Control Interface** | `Streamlit` & `Plotly` | Renders a dark glassmorphic, interactive 2.5D floor plan and a chatbot terminal. |

---

## 🔄 End-to-End Processing Architecture

The data flows asynchronously across the platform through parallel background processing threads (`asyncio`), ensuring real-time multi-agent intelligence without degrading live camera frame rates:

```
[ bridge.py Ingestion ] ──► Pushes Dynamic (X, Y) Telemetry Rows (30 FPS)
│
▼
[ main.py FastAPI Core ] ─► Appends Data to Telemetry Sliders (TRACKING_RAM_BUFFER)
│
┌──────────────────────────┼──────────────────────────┐
▼ (Parallel Workers)       ▼ (Parallel Workers)       ▼ (Parallel Workers)
[ Movement Agent ]          [ Anomaly Agent ]          [ Crowd Agent ]
Amazon Chronos-Bolt         Kinematic Sequence         CSRNet Dilated-CNN
Generates 5s Vectors        Calculates Tortuosity      Measures dDensity/dt
│                          │                          │
└──────────────────────────┼──────────────────────────┘
│
▼
[ NetworkX DB Sync ] ─────► Maps Interpersonal Connections & Louvain Social Cells
│
▼ Broadcasts Payloads via WebSockets
[ mvp_dashboard.py Streamlit UI ] ──► Renders Live Conversational Chatbot & Interactive Maps
```

---

## 💼 Strategic Real-World Use Cases

The URG-IS multi-agent layer translates raw matrix parameters into automated solutions for high-stakes enterprise scenarios:

### 1. Public Welfare: Missing Child & Person Tracing
* **The Problem:** Reconstructing routes and finding a missing subject across fragmented cameras manually is highly time-consuming.
* **The Solution:** The system parses historical coordinate ticks directly through the local **Amazon Chronos-Bolt Transformer Model**, drawing a futuristic dotted trajectory trail 5 seconds into the future so field officers can intercept the target. Simultaneously, it sweeps proximity graph edges to pinpoint exactly who they were last seen interacting with.

### 2. Smart City: Crowd Safety & Stampede Prevention
* **The Problem:** Sudden human accumulation bottlenecks create life-threatening compression forces before operators visually notice a crowd.
* **The Solution:** The **Crowd Safety Agent** maps raw pixels into spatial density arrays via `CSRNet`, evaluating the **Rate of Density Acceleration ($\frac{\partial\text{Density}}{\partial t}$)** over time. If a flash crowd surge breaches safety thresholds, it immediately triggers automated software-driven overrides to release physical exit gates.

### 3. High-Stakes Security: Coordinated Threat Network Mapping
* **The Problem:** Isolating bad actors, scouts, or sleeper cells hiding inside standard civilian crowds.
* **The Solution:** The **Behavioral Anomaly Agent** measures a path's straightness and variance to isolate individuals walking in suspicious loitering loops. Once a threat score flags a warning, the system queries the **NetworkX relationship graph**, runs **Louvain Clustering**, and highlights their hidden network associates on screen without relying on invasive external facial recognition APIs.

### 4. Civic Order: Smart Governance & Traffic Violations
* **The Problem:** Enforcing civil laws (like littering or public loitering) and traffic rules (like triple-riding or helmet gaps).
* **The Solution:** The **Civic Order Agent** tracks objects dropped by individuals (detecting object-permanence separation) or monitors vehicle occupancy bounding boxes. It links these urban infractions back to the specific citizen tracking ID inside the graph engine, transforming a routine fine into an active behavioral pattern tracker.

---

## 🚀 Air-Gapped Deployment & Execution

Because the architecture forces `local_files_only=True` and toggles environmental offline variables, the complete system runs fully operational in a **strictly air-gapped environment** with **0% network leak risk**.

### Prerequisites
Ensure your localized virtual environment is active and python modules are installed:
```bash
source reid_env/bin/activate
```

### Deployed Launch Procedure

To run the complete production system end-to-end using your local data frames, open **three independent terminal windows** and execute the following scripts in order:

1. **Terminal 1: Start the Central Orchestrator Backend Core**
```bash
python main.py
```
*Loads the foundational models directly into RAM and activates the FastAPI gateway on port 8000.*

2. **Terminal 2: Launch the Interactive Control Dashboard**
```bash
streamlit run mvp_dashboard.py
```
*Spins up the premium dark glassmorphic UI layout and establishes a live network socket client connection.*

3. **Terminal 3: Fire the Live Dataset Telemetry Bridge**
```bash
python bridge.py
```
*Streams real, sequential coordinate logs directly from your local dataset files at a steady 30 FPS framework rate.*

---

*Developed by Bhargavi on MacBook Air Architecture — Sovereign AI by Design.*
