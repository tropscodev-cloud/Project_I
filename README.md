# Universal Relationship Graph Intelligence System (URG-IS)

"Intelligence is not about capturing the past; it is about automating the anticipation of the future." (URG-IS Deployed Core Principle)

It is not a surveillance tool; it is a Relationship Intelligence Engine. It takes raw video and automatically transforms it into a living social network. By creating a unique digital signature for every person, URG-IS maps human behavior in real-time. We have moved the needle from Passive Monitoring to Predictive Intelligence.

## Complete Technical Stack

| Pipeline Layer | Technology / Model Implemented | System Function |
| :--- | :--- | :--- |
| Ingestion Engine | OpenCV and FFmpeg | Captures live camera video streams and processes them frame by frame. |
| Object Detection | Ultralytics YOLOv8 (yolov8s.pt) | Fast scanning of each frame to detect people and vehicles. |
| Intra-Cam Tracking | BoT-SORT | Keeps tracking the same person across different frames. |
| Global Re-ID | OSNet (TorchReID) and FAISS | Creates a unique digital fingerprint to identify the same person across different cameras. |
| Sovereign Database | NetworkX | A local database that maps how different people are connected to each other. |
| Predictive Foundation | amazon/chronos-bolt-tiny | A model that runs locally to forecast where a person will walk next. |
| Cognition Layer | Ollama (Llama 3.2 3B Instruct) | Converts data relationships into simple, readable text reports. |
| Control Interface | Streamlit and Plotly | A clean and dark dashboard showing a floor plan and a chat screen. |

## Pretrained Hugging Face Models in Use

The system uses the following pretrained models from the Hugging Face Hub:
* amazon/chronos-bolt-tiny (Used locally to predict future walking paths of tracked people by looking at their past movements)

## Processing Architecture

The system processes data in three main steps:

1. Data Ingestion: The file bridge.py captures video frames and extracts the location coordinates of people at a rate of 30 frames per second.
2. Central Server: The file main.py receives the location data and updates the temporary tracking memory buffer.
3. Live Analysis: The data is immediately analyzed by three specialized tracking modules:
   * Movement Module: Predicts where the person is likely to walk in the next 5 seconds.
   * Anomaly Module: Evaluates if a person is walking in suspicious loops.
   * Crowd Module: Calculates crowd densities and flags if a pathway is getting dangerously blocked.

Once the analysis is complete, the results are sent to a local database to map connections and update the Streamlit dashboard in real time.

## Strategic Real-World Use Cases

The system processes information to solve the following situations:

1. Public Welfare: Tracking Missing Persons
   * The Problem: Reconstructing routes to find a lost person across multiple cameras manually is slow and difficult.
   * The Solution: The system analyzes past movement data to draw a predicted path five seconds into the future, showing security staff where the person is likely to go. It also checks who they were last seen walking with.

2. Smart City: Crowd Safety and Flow Management
   * The Problem: Sudden crowd build-ups can create dangerous situations before operators notice them on screen.
   * The Solution: The system calculates the rate at which a crowd is growing. If the crowd density increases too quickly or exceeds safety limits, the system sends alerts to open emergency exit gates automatically.

3. High-Stakes Security: Loitering and Behavior Pattern Analysis
   * The Problem: Spotting suspicious individuals or coordinate loitering behaviors in large crowds.
   * The Solution: The system monitors walking paths to detect loitering. If a high threat level is detected, the database checks if they have close connections with other suspicious profiles to flag them on the screen.

4. Civic Order: Smart City Traffic and Rules
   * The Problem: Enforcing city rules such as littering and traffic violations like riding a motorcycle without a helmet or with too many passengers.
   * The Solution: The system monitors vehicle occupants and detects if anyone drops trash. It logs these events to the tracking database for processing.


### Prerequisites

Activate your local python environment and install the required modules:
```bash
source reid_env/bin/activate
```

### Deployed Launch Procedure

To run the system, open three separate terminal windows and run the scripts in this order:

1. Terminal 1: Start the Backend Orchestrator
```bash
python main.py
```
This loads the models into memory and starts the local server.

2. Terminal 2: Launch the Control Dashboard
```bash
streamlit run mvp_dashboard.py
```
This opens the web interface in your browser.

3. Terminal 3: Run the Telemetry Bridge
```bash
python bridge.py
```
This feeds the tracking data to the backend server.
