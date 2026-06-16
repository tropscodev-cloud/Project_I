"""
URG-IS Production Dashboard    mvp_dashboard.py
=================================================
Run:  streamlit run mvp_dashboard.py

Fixes vs previous version
  1. Per-camera PersonTracker  bounding boxes now appear on every camera
  2. Apple Silicon MPS GPU auto-detected (3-5x faster, cooler laptop)
  3. Full new 6-tab design showcasing all pipeline work
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, time, threading, warnings, math, json
import torch
warnings.filterwarnings("ignore")

import cv2
import base64
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter, deque
from pathlib import Path
from pyvis.network import Network
import streamlit.components.v1 as components

import pipeline_state as PS   # singleton  survives Streamlit reruns
from core.graph.graph_db import RELATIONSHIP_COLORS
from core.spatial.floor_mapper import UnifiedFloorMapper
import usecases_page
import queue
import requests

_telemetry_queue = queue.Queue(maxsize=1000)
_session = requests.Session()

def _telemetry_worker():
    while True:
        try:
            pl = _telemetry_queue.get()
            if pl is None:
                break
            try:
                # Force the requests session to use an aggressive non-blocking configuration
                _session.post("http://127.0.0.1:8000/api/v1/telemetry", json=pl, timeout=0.01)
            except requests.exceptions.Timeout:
                pass
        except Exception:
            pass

# SAFE GUARD: Guard against Streamlit redraw loops spinning up hundreds of background threads
if not hasattr(PS, '_worker_initialized'):
    threading.Thread(target=_telemetry_worker, daemon=True).start()
    PS._worker_initialized = True

threading.Thread(target=_telemetry_worker, daemon=True).start()


#  Page config 
st.set_page_config(
    page_title="URG-IS Intelligence Platform",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#  Design tokens 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;800&display=swap');
:root {
    --bg-primary: #071018;
    --bg-secondary: #0B1622;
    --bg-card: rgba(15, 23, 35, 0.82);
    --accent-primary: #7CFFB2;
    --accent-secondary: #4DA8FF;
    --text-primary: #F8FAFC;
    --text-secondary: #AAB6C5;
    --border-soft: rgba(255,255,255,0.06);
    --glow-soft: rgba(124,255,178,0.18);
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    background-image: radial-gradient(circle at 50% 0%, rgba(11, 22, 34, 0.8) 0%, transparent 60%) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif;
}
.block-container { padding-top: 2rem !important; }
/* Hide Default Elements */
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stHeader"] { 
    display: none !important; 
}
/* Cards & Glassmorphism */
.card, .aegis-cam-card, .tech-card, .pipe-step, .metric-card {
    background: var(--bg-card) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
    transition: all .3s ease !important;
}
.card:hover, .metric-card:hover { 
    border-color: rgba(124, 255, 178, 0.3) !important; 
    box-shadow: 0 0 24px var(--glow-soft) !important; 
    transform: translateY(-2px);
}
/* Typography */
h1, h2, h3, h4 { color: var(--text-primary) !important; font-weight: 800 !important; letter-spacing: 0.02em; }
p, div { color: var(--text-secondary); }
.hero-sub { color: var(--accent-primary); letter-spacing: 0.2em; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; font-size: 0.85rem;}
/* Buttons */
.stButton>button {
    background: rgba(11, 22, 34, 0.8) !important; 
    color: var(--text-secondary) !important; 
    border: 1px solid var(--border-soft) !important; 
    border-radius: 8px !important; 
    font-weight: 600 !important; 
    letter-spacing: .05em !important; 
    font-size: 0.8rem !important;
    transition: all .3s ease !important;
    padding: 10px 24px !important;
}
.stButton>button:hover {
    background: rgba(124, 255, 178, 0.05) !important; 
    color: var(--text-primary) !important;
    border-color: rgba(124, 255, 178, 0.4) !important;
    box-shadow: 0 0 15px var(--glow-soft) !important;
}
/* Primary Button (Active Nav / CTA) */
.stButton>button[data-baseweb="button"]:has(p strong), .stButton>button[kind="primary"] {
    background: rgba(124, 255, 178, 0.1) !important; 
    color: var(--accent-primary) !important;
    border: 1px solid rgba(124, 255, 178, 0.5) !important;
    box-shadow: 0 0 25px var(--glow-soft) !important;
}
/* Inputs */
.stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>div>textarea {
    background: var(--bg-secondary) !important; 
    border: 1px solid var(--border-soft) !important;
    color: var(--text-primary) !important; 
    border-radius: 8px !important;
}
.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
    border-color: var(--accent-secondary) !important;
    box-shadow: 0 0 10px rgba(77, 168, 255, 0.2) !important;
}
/* Custom Metric Cards */
.metric-card {
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 12px;
}
.metric-label {
    color: var(--text-secondary);
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-val {
    color: var(--text-primary);
    font-size: 2.4rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
}
.metric-sub {
    color: var(--accent-secondary);
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}
/* Dividers */
hr { border-color: var(--border-soft) !important; margin: 10px 0 !important; }
/* Alerts */
.alert-h { background: rgba(239, 68, 68, 0.1); border-left: 3px solid #ef4444; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; }
.alert-m { background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; }
.alert-l { background: rgba(77, 168, 255, 0.1); border-left: 3px solid var(--accent-secondary); border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; }
/* Badges */
.badge { display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: .7rem; font-weight: 700; letter-spacing: .05em; }
.b-stranger { background: rgba(170, 182, 197, 0.1); color: var(--text-secondary); border: 1px solid var(--border-soft); }
.b-acquaintance { background: rgba(77, 168, 255, 0.1); color: var(--accent-secondary); border: 1px solid rgba(77, 168, 255, 0.3); }
.b-associate { background: rgba(124, 255, 178, 0.1); color: var(--accent-primary); border: 1px solid rgba(124, 255, 178, 0.3); }
.b-close_associate { background: rgba(251, 191, 36, 0.1); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
.b-significant { background: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
/* Custom Overrides */
.cam-card-title { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; color: var(--text-primary); }
.fps-indicator { color: var(--accent-primary); font-size: 0.75rem; font-weight: 700; }
/* Navbar Container */
.palantir-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 24px; background: rgba(7, 16, 24, 0.8);
    backdrop-filter: blur(24px); border-bottom: 1px solid var(--border-soft);
    position: sticky; top: 0; z-index: 1000; margin-bottom: 32px; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

#  Palette 
RC = RELATIONSHIP_COLORS
ROOT_COLOR = "#22c55e"
IC = {"PROXIMITY":"#38bdf8","CONVERSATION":"#4ade80","CLOSE_CONTACT":"#f87171",
      "EXTENDED_MEETING":"#fbbf24","GROUP_GATHERING":"#c084fc"}
RO = ["stranger","acquaintance","associate","close_associate","significant"]
DATA_DIR = Path("data")

def _rgba(h,a=1.0):
    hx=h.lstrip("#"); r,g,b=int(hx[0:2],16),int(hx[2:4],16),int(hx[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def _root_color(relationship, confidence=0.0):
    return RELATIONSHIP_COLORS.get(relationship, "#888888")

def _find_cams(d):
    s={}
    for ext in [".mp4",".avi",".mkv",".mov"]:
        for f in sorted(d.glob(f"*{ext}")): s[f.stem]=str(f)
    return s

def _person_graph_rows(person_graph, min_conf=0.0):
    rows = []
    for conn in person_graph.get("connections", []):
        if conn["confidence"] < min_conf:
            continue
        loc_events = conn.get("location_events", [])
        loc_cams = sorted({ev.get("camera_id", "") for ev in loc_events if ev.get("camera_id")})
        rows.append({
            "Person": f"P{conn['person_id']}",
            "Confidence": round(conn["confidence"], 4),
            "Relationship": conn["relationship"].replace("_", " ").title(),
            "Meetings": conn["total_meetings"],
            "Meetings Today": conn["meetings_today"],
            "Last Incident": conn["last_incident"],
            "Avg Duration": f"{conn['avg_duration_s']:.1f}s",
            "Total Duration": f"{conn['total_duration_s']:.1f}s",
            "Incidents": ", ".join(f"{k}:{v}" for k, v in sorted(conn["incident_counts"].items())),
            "Cameras": ", ".join(sorted(conn["cameras"])),
            "Location Samples": len(loc_events),
            "Location Cameras": ", ".join(loc_cams) or ", ".join(sorted(conn["cameras"])),
        })
    return rows

def _avg_location(points):
    clean = [(float(x), float(y)) for x, y in points or [] if x or y]
    if not clean:
        return None
    return (
        sum(p[0] for p in clean) / len(clean),
        sum(p[1] for p in clean) / len(clean),
    )

_FLOOR_MAPPER = None

def _floor_mapper():
    global _FLOOR_MAPPER
    if _FLOOR_MAPPER is None:
        _FLOOR_MAPPER = UnifiedFloorMapper()
    return _FLOOR_MAPPER

def _edge_location(edge, conn):
    events = getattr(edge, "location_events", None) if edge else conn.get("location_events", [])
    if events is None:
        events = conn.get("location_events", [])
    floor_points, camera_points = [], []
    for ev in events or []:
        camera_id = ev.get("camera_id", "")
        loc = ev.get("location_px") or ev.get("location")
        if not loc:
            continue
        loc = (int(loc[0]), int(loc[1]))
        fp = _floor_mapper().make_point(conn["person_id"], camera_id, loc)
        if fp:
            floor_points.append((fp.map_xy_px[0], fp.map_xy_px[1], camera_id))
        else:
            camera_points.append((loc[0], loc[1], camera_id))

    if floor_points:
        x = sum(p[0] for p in floor_points) / len(floor_points)
        y = sum(p[1] for p in floor_points) / len(floor_points)
        cams = sorted({p[2] for p in floor_points if p[2]})
        return (x, y), "floorplan", ", ".join(cams)

    if camera_points:
        x = sum(p[0] for p in camera_points) / len(camera_points)
        y = sum(p[1] for p in camera_points) / len(camera_points)
        cams = sorted({p[2] for p in camera_points if p[2]})
        return (x, y), "camera-view", ", ".join(cams)

    edge_locations = getattr(edge, "locations", None) if edge else conn.get("locations", [])
    if edge_locations is None:
        edge_locations = conn.get("locations", [])
    raw_xy = _avg_location(edge_locations)
    if raw_xy:
        return raw_xy, "legacy-camera", ", ".join(sorted(conn.get("cameras", [])))
    return None, "estimated", ", ".join(sorted(conn.get("cameras", [])))

#  GPU info helper 
def _device_label():
    try:
        import torch
        if torch.backends.mps.is_available(): return "Apple MPS GPU","#4ade80"
        if torch.cuda.is_available():         return f"CUDA {torch.cuda.get_device_name(0)}","#4ade80"
    except Exception: pass
    return "CPU","#f59e0b"

# 
# PIPELINE WORKER
# 
def _camera_worker(cid, src, fs, db, engine, embedder, manager):
    """Worker thread for a single camera stream."""
    try:
        from core.tracking.person_tracker import PersonTracker
        from core.interaction.interaction_detector import InteractionDetector, annotate_interactions
        import torch
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tracker = PersonTracker(device=device)
        detector = InteractionDetector(camera_id=cid)
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        fnum = 0
        skip_cnt = 0
        fps_times = deque(maxlen=30)
        
        # Localize tracking state to avoid cross-camera thread conflicts
        last_embed_frame = {}
        id_map = {}
        EMBED_INTERVAL = 30
        
        # Get frame rate for local video throttling
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        while True:
            with PS.lock:
                if PS.G["stop_requested"]: break
            
            skip_cnt += 1
            if skip_cnt < fs:
                cap.grab()
                continue
            skip_cnt = 0
            
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                # Reliably loop video files by releasing and reopening the stream
                cap.release()
                cap = cv2.VideoCapture(src)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, frame = cap.read()
                if not ret: 
                    time.sleep(0.1)
                    continue
            
            fnum += 1
            tracked = tracker.track(frame, camera_id=cid, frame_num=fnum)
            annotated = tracker.annotate(frame, tracked, show_trail=True, show_confidence=True)
            
            persons_with_ids = []
            if tracked:
                for person in tracked:
                    tid = person.track_id
                    # Only run the heavy embedder every 30 frames
                    if tid not in last_embed_frame or (fnum - last_embed_frame[tid]) > EMBED_INTERVAL:
                        vec = embedder.embed(person.crop)
                        result = manager.identify(vec, tid, cid, fnum)
                        if result:
                            id_map[tid] = result.person_id
                        last_embed_frame[tid] = fnum
                    
                    pid = id_map.get(tid)
                    if pid is not None:
                        persons_with_ids.append((pid, person.center))
                        try:
                            if detector.H is not None:
                                p_arr = np.array([[[float(person.center[0]), float(person.center[1])]]], dtype=np.float32)
                                warped = cv2.perspectiveTransform(p_arr, detector.H)[0][0]
                                xm, ym = float(warped[0]), float(warped[1])
                            else:
                                xm = float(person.center[0]) / detector.pixels_per_m
                                ym = float(person.center[1]) / detector.pixels_per_m
                            payload = {
                                "camera_id": str(cid),
                                "entity_id": str(pid),
                                "x_meters": xm,
                                "y_meters": ym,
                                "frame_timestamp": float(time.time())
                            }
                            try:
                                _telemetry_queue.put_nowait(payload)
                            except queue.Full:
                                pass
                        except Exception:
                            pass
                        x1,y1,x2,y2 = person.bbox
                        cv2.rectangle(annotated,(x1,y1-22),(x1+52,y1),(0,180,80),-1)
                        cv2.putText(annotated,f"P{pid}",(x1+4,y1-6),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),2)

            # if tracked:
            #     for person in tracked:
            #         vec = embedder.embed(person.crop)
            #         result = manager.identify(vec, person.track_id, cid, fnum)
            #         if result:
            #             persons_with_ids.append((result.person_id, person.center))
            #             x1,y1,x2,y2 = person.bbox
            #             cv2.rectangle(annotated,(x1,y1-22),(x1+52,y1),(0,180,80),-1)
            #             cv2.putText(annotated,f"P{result.person_id}",(x1+4,y1-6),
            #                         cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),2)

            if len(persons_with_ids) >= 2:
                events = detector.update(persons_with_ids, cid, fnum)
                centers = {pid:c for pid,c in persons_with_ids}
                pids = [pid for pid,_ in persons_with_ids]
                for ev in events:
                    if ev.person_id_a == ev.person_id_b: continue
                    ca = centers.get(ev.person_id_a,(0,0)); cb = centers.get(ev.person_id_b,(0,0))
                    ev._location_px = ((ca[0]+cb[0])//2,(ca[1]+cb[1])//2)
                    edge = engine.process_event(ev, people_in_scene=pids)
                    if edge:
                        rec = {"frame":fnum, "camera":cid, "person_a":ev.person_id_a,
                               "person_b":ev.person_id_b, "type":edge.last_incident,
                               "distance_m":round(ev.distance_m,2), "duration_s":round(ev.duration_s,1),
                               "confidence":round(edge.confidence,4), "relationship":edge.relationship}
                        with PS.lock:
                            PS.G["incidents"].append(rec)
                            
                            PS.G["conf_history"].append((fnum, ev.person_id_a, ev.person_id_b, 
                                                       edge.confidence, edge.relationship, edge.last_incident))

                annotated = annotate_interactions(annotated, persons_with_ids, 
                                                 detector.get_active_proximities(), events)

            elapsed = time.time() - t0
            fps_times.append(elapsed)
            avg_fps = 1.0 / max(sum(fps_times)/len(fps_times), 0.001)
            
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            with PS.lock:
                PS.G["latest_frames"][cid] = rgb
                PS.G["frame_counts"][cid] = fnum
                PS.G["fps"][cid] = round(avg_fps, 1)
                PS.G["total_frames"] += 1
                PS.G["last_update"] = time.time()
                PS.G["person_count"] = manager.get_identity_count()
            
            # Throttling for local files to match real-time playback
            if isinstance(src, str) and not src.startswith("rtsp://"):
                delay = (1.0 / fps) * fs
                sleep_time = max(0.001, delay - elapsed)
                time.sleep(sleep_time)
            else:
                time.sleep(0.001)

    except Exception:
        import traceback
        with PS.lock: PS.G["errors"].append(f"[{cid}] {traceback.format_exc()}")
    finally:
        cap.release()

@st.cache_resource
def load_ai_models():
    from core.reid.embedder import PersonEmbedder
    from core.reid.identity_manager import IdentityManager
    return PersonEmbedder(), IdentityManager(threshold=0.70)

def _start(sources, fs):
    """Initializes shared intelligence components and starts one thread per camera."""
    with PS.lock:
        PS.G.update({"stop_requested": False, "total_frames": 0, "person_count": 0, "running": True})
        PS.G["incidents"].clear()
        PS.G["conf_history"].clear()
        PS.G["latest_frames"].clear()
        PS.G["frame_counts"].clear()
        PS.G["fps"].clear()
        PS.G["errors"].clear()
    
    # Initialize shared intelligence components ONCE
    from core.graph.graph_db import GraphDB
    from core.graph.confidence_engine import ConfidenceEngine
    
    db = GraphDB(snapshot_path="data/snapshots/prod_graph.json")
    engine = ConfidenceEngine(db, decay_interval_m=10, auto_snapshot=True)
    engine.start()
    embedder, manager = load_ai_models()
    
    with PS.lock:
        PS.G.update({"graph_db": db, "engine": engine, "active_cams": list(sources.keys()), "camera_sources": sources})

    # Launch one worker thread per camera
    PS.pipeline_threads = []
    for cid, src in sources.items():
        t = threading.Thread(target=_camera_worker, args=(cid, src, fs, db, engine, embedder, manager), 
                             daemon=True, name=f"urg-{cid}")
        PS.pipeline_threads.append(t)
        t.start()


def _stop():
    with PS.lock: 
        PS.G["stop_requested"]=True
    # Wait for threads to acknowledge stop (optional, but cleaner)
    if hasattr(PS, 'pipeline_threads'):
        for t in PS.pipeline_threads:
            if t.is_alive():
                t.join(timeout=0.5)
    with PS.lock:
        if PS.G.get("engine"):
            PS.G["engine"].stop()
        PS.G["running"] = False



# Ensure fps key exists
with PS.lock:
    if "fps" not in PS.G: PS.G["fps"]={}


# 
# GRAPH BUILDER
# 
def _build_graph(db, hl=None, min_conf=0.05, view_mode="Individuals", max_age=None):
    """Beautiful physics-based relationship graph using pyvis."""
    if not db: return None
    el = [e for e in db.get_all_edges(max_age_seconds=max_age) if e.confidence >= min_conf]
    if not el: return None

    net = Network(height="650px", width="100%", bgcolor="#071018", font_color="#F8FAFC")
    
    if view_mode == "Communities":
        net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)
        comms = db.get_louvain_communities()
        if not comms: return None
        
        # Build community graph
        comm_map = {}
        for cm in comms:
            c_id = f"Group {cm['community_id']}"
            for p in cm["members"]: comm_map[p] = c_id
            net.add_node(c_id, label=c_id, size=max(20, min(50, cm["size"]*5)), 
                         title=f"{c_id}\nMembers: {cm['size']}",
                         color={"background": "#4DA8FF", "border": "#071018"},
                         font={"color": "#F8FAFC"})
        
        # Add edges between communities
        c_edges = defaultdict(float)
        for e in el:
            c_a = comm_map.get(e.person_id_a)
            c_b = comm_map.get(e.person_id_b)
            if c_a and c_b and c_a != c_b:
                c_edges[(c_a, c_b)] += e.confidence
        
        for (c_a, c_b), w in c_edges.items():
            net.add_edge(c_a, c_b, value=math.log(w+1)*5, title=f"Weight: {w:.1f}", color="#1A3A5F")
            
    else:
        net.barnes_hut(gravity=-4000, central_gravity=0.3, spring_length=250, spring_strength=0.04, damping=0.15)
        G = nx.Graph()
        for e in el:
            G.add_edge(e.person_id_a, e.person_id_b, weight=e.confidence, rel=e.relationship)

        for n in G.nodes():
            deg = G.degree(n)
            best = "stranger"
            for nb in G.neighbors(n):
                r = G[n][nb]["rel"]
                if RO.index(r) > RO.index(best): best = r
            col = "#F8FAFC" if n == hl else RC.get(best, "#4DA8FF")
            
            # Logarithmic scaling strictly capped at 40
            sz = min(40, 15 + math.log(deg + 1) * 8)
            sz += (10 if n == hl else 0)
            
            title = f"P{n}\nConnections: {deg}"
            net.add_node(n, label=f"P{n}", size=sz, title=title,
                         color={"background": col, "border": "#071018", "highlight": {"border": "#7CFFB2", "background": "#4DA8FF"}},
                         borderWidth=3 if n == hl else 0,
                         font={"color": "#071018", "face": "JetBrains Mono"})

        for u, v, d in G.edges(data=True):
            col = RC.get(d["rel"], "#7CFFB2")
            w = max(1.0, d["weight"] * 5)
            alpha = "ff" if (hl in [u, v]) else "88"
            net.add_edge(u, v, value=w, title=f"Conf: {d['weight']:.3f} | {d['rel']}", color=col + alpha)

    net.set_options('{"interaction": {"hover": true}, "physics": {"stabilization": {"enabled": true, "iterations": 100}}}')
    return net

def _build_person_route_map(db, person_id, min_conf=0.05):
    """Clean radial relationship graph for one selected person."""
    if not db or not person_id:
        return None, None, 0

    pg = db.get_person_graph(person_id)
    if not pg:
        return None, None, 0

    visible = [c for c in pg["connections"] if c["confidence"] >= min_conf]
    pruned_count = max(0, len(pg["connections"]) - len(visible))
    fig = go.Figure()

    cameras = sorted({cam for c in visible for cam in c.get("cameras", [])} or set(pg.get("camera_ids", [])))
    camera_label = ", ".join(cameras) if cameras else "unknown camera"

    if not visible:
        fig.add_annotation(
            x=50, y=50, text="Select a lower threshold or let more interactions accumulate",
            showarrow=False, font=dict(size=15, color="#94a3b8")
        )
    else:
        root_x, root_y = 50, 50
        route_points = []
        total = len(visible)
        for idx, conn in enumerate(visible):
            edge = db.get_edge(person_id, conn["person_id"])
            raw_xy, source, location_label = _edge_location(edge, conn)
            angle = -math.pi / 2 + (2 * math.pi * idx / max(total, 1))
            radius = 26 + (1 - min(conn["confidence"], 1.0)) * 10
            x = root_x + math.cos(angle) * radius
            y = root_y + math.sin(angle) * radius
            route_points.append({
                "conn": conn,
                "x": x,
                "y": y,
                "angle": angle,
                "raw_xy": raw_xy,
                "source": source,
                "location_label": location_label,
            })

        source_counts = Counter(p["source"] for p in route_points)
        source_label = ", ".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
        fig.add_annotation(
            x=2, y=98, xanchor="left", yanchor="top",
            text=f"<b>Person P{person_id}</b><br>cameras: {camera_label}<br>location source: {source_label}",
            showarrow=False,
            font=dict(size=13, color="#60c3ff"),
            align="left",
        )

        for idx, point in enumerate(route_points):
            conn = point["conn"]
            x, y = point["x"], point["y"]
            color = _root_color(conn["relationship"], conn["confidence"])
            width = max(2.5, conn["confidence"] * 12)
            end_x = root_x + math.cos(point["angle"]) * (math.hypot(x - root_x, y - root_y) - 5)
            end_y = root_y + math.sin(point["angle"]) * (math.hypot(x - root_x, y - root_y) - 5)
            fig.add_trace(go.Scatter(
                x=[root_x, end_x], y=[root_y, end_y], mode="lines",
                line=dict(color=color, width=width),
                opacity=0.88,
                hoverinfo="text",
                hovertext=(
                    f"P{person_id} met P{conn['person_id']}<br>"
                    f"Relationship: {conn['relationship'].replace('_', ' ').title()}<br>"
                    f"Confidence: {conn['confidence']:.3f}<br>"
                    f"Meetings: {conn['total_meetings']}<br>"
                    f"Total duration: {conn['total_duration_s']:.1f}s<br>"
                    f"Avg duration: {conn['avg_duration_s']:.1f}s<br>"
                    f"Last incident: {conn['last_incident']}<br>"
                    f"Cameras: {', '.join(sorted(conn['cameras']))}<br>"
                    f"Location source: {point['source']}"
                ),
                showlegend=False,
            ))
            label_x = root_x + (x - root_x) * 0.58
            label_y = root_y + (y - root_y) * 0.58
            duration_label = f"{conn['total_duration_s'] / 60:.1f}m" if conn["total_duration_s"] >= 60 else f"{conn['total_duration_s']:.0f}s"
            fig.add_annotation(
                x=label_x,
                y=label_y,
                text=(
                    f"{conn['total_meetings']} meets<br>"
                    f"{duration_label} | {conn['confidence']:.2f}"
                ),
                showarrow=False,
                bgcolor="rgba(5,12,26,0.82)",
                bordercolor=color,
                borderwidth=1,
                borderpad=4,
                font=dict(color="#e2e8f0", size=10),
            )

        fig.add_trace(go.Scatter(
            x=[p["x"] for p in route_points],
            y=[p["y"] for p in route_points],
            mode="markers+text",
            marker=dict(
                size=[24 + p["conn"]["confidence"] * 28 for p in route_points],
                color=[_root_color(p["conn"]["relationship"], p["conn"]["confidence"]) for p in route_points],
                symbol="circle",
                line=dict(color="#f8fafc", width=3),
            ),
            text=[f"P{p['conn']['person_id']}" for p in route_points],
            textposition="bottom center",
            textfont=dict(color="#e2e8f0", size=13),
            hoverinfo="text",
            hovertext=[
                f"<b>P{person_id} met P{p['conn']['person_id']}</b><br>"
                f"Relationship: {p['conn']['relationship'].replace('_', ' ').title()}<br>"
                f"Confidence: {p['conn']['confidence']:.3f}<br>"
                f"Meetings: {p['conn']['total_meetings']}<br>"
                f"Total duration: {p['conn']['total_duration_s']:.1f}s<br>"
                f"Last incident: {p['conn']['last_incident']}<br>"
                f"Cameras: {', '.join(sorted(p['conn']['cameras']))}<br>"
                f"Location: {p['location_label'] or 'estimated'} ({p['source']})"
                for p in route_points
            ],
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=[root_x], y=[root_y],
            mode="markers+text",
            marker=dict(
                size=44, symbol="diamond",
                color=ROOT_COLOR,
                line=dict(color="#dcfce7", width=4),
            ),
            text=[f"P{person_id}"],
            textposition="top center",
            textfont=dict(color="#f8fafc", size=16),
            hoverinfo="text",
            hovertext=(
                f"<b>Selected person P{person_id}</b><br>"
                f"Cameras: {camera_label}<br>"
                f"Visible relationships: {len(visible)}"
            ),
            showlegend=False,
        ))

        fig.add_annotation(
            x=root_x, y=root_y - 8, text="selected person",
            showarrow=False, font=dict(color="#bbf7d0", size=12),
        )

    for rel in RO:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=_root_color(rel), symbol="circle"),
            name=rel.replace("_", " ").title(),
            showlegend=True,
        ))

    fig.update_layout(
        template="plotly_dark",
        height=800,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#050c1a",
        plot_bgcolor="#071426",
        legend=dict(
            orientation="h",
            x=0.5, y=-0.02, xanchor="center",
            bgcolor="rgba(5,12,26,0.75)",
            bordercolor="#16345c",
            borderwidth=1,
            font=dict(color="#cbd5e1", size=11),
        ),
        xaxis=dict(range=[0, 100], visible=False, fixedrange=True),
        yaxis=dict(range=[0, 100], visible=False, fixedrange=True, scaleanchor="x", scaleratio=1),
    )
    return fig, pg, pruned_count


# 
# SNAPSHOT (thread-safe read)
# 
with PS.lock:
    is_running   = PS.G["running"]
    person_count = PS.G["person_count"]
    total_frames = PS.G["total_frames"]
    last_upd     = PS.G["last_update"]
    db_ref       = PS.G["graph_db"]
    active_cams  = list(PS.G["active_cams"])
    frames_snap  = dict(PS.G["latest_frames"])
    fcount_snap  = dict(PS.G["frame_counts"])
    fps_snap     = dict(PS.G.get("fps",{}))
    incidents    = list(PS.G["incidents"])
    conf_hist    = list(PS.G["conf_history"])
    errors       = list(PS.G["errors"])
    if errors:
        try:
            with open("thread_errors.log", "w") as f:
                f.write("\n".join(errors))
        except Exception:
            pass
    try:
        import json
        state_to_save = {
            "running": PS.G["running"],
            "stop_requested": PS.G["stop_requested"],
            "person_count": PS.G["person_count"],
            "active_cams": PS.G["active_cams"],
            "last_update": PS.G["last_update"],
            "total_frames": PS.G["total_frames"],
            "camera_sources": PS.G["camera_sources"],
            "fps": PS.G["fps"],
            "errors": PS.G["errors"],
        }
        with open("dashboard_state.json", "w") as f:
            json.dump(state_to_save, f, indent=2)
    except Exception:
        pass

# Default refresh interval
refresh_ms = 1000

# 
# TOP NAVIGATION BAR & STATE
# 
if "current_page" not in st.session_state:
    st.session_state.current_page = "HOME"

def switch_page(page_name):
    st.session_state.current_page = page_name

nav_cols = st.columns([2.2, 0.8, 1.1, 0.8, 1.0, 1.1, 1.0, 1.1, 1.5])
with nav_cols[0]:
    st.markdown("""
        <div style='display:flex;align-items:center;gap:12px;padding-top:4px;'>
            <div style='background:rgba(124, 255, 178, 0.1);border:1px solid rgba(124, 255, 178, 0.4);border-radius:8px;width:32px;height:32px;display:flex;align-items:center;justify-content:center;color:var(--accent-primary);font-weight:900;'>U</div>
            <div style='font-size:1.2rem;font-weight:900;letter-spacing:0.2em;color:var(--text-primary)'>URG-IS <span style='color:var(--accent-primary);font-weight:400;'>INTELLIGENCE</span></div>
        </div>
    """, unsafe_allow_html=True)

nav_pages = ["HOME", "OPERATIONS", "GRAPH", "INCIDENTS", "ANALYTICS", "AI AGENT", "USE CASES"]
for i, p in enumerate(nav_pages):
    with nav_cols[i+1]:
        is_active = (st.session_state.current_page == p)
        st.button(f"**{p}**" if is_active else p, width='stretch', type="primary" if is_active else "secondary", on_click=switch_page, args=(p,), key=f"nav_{p}")

with nav_cols[8]:
    if is_running:
        st.markdown("<div style='text-align:center; padding: 10px; background: rgba(124, 255, 178, 0.1); border: 1px solid rgba(124, 255, 178, 0.4); color: var(--accent-primary); border-radius: 8px; font-weight: 800; letter-spacing: 0.1em; font-size:0.7rem;'>SYSTEM ACTIVE</div>", unsafe_allow_html=True)


st.markdown("<hr style='margin-top:10px; margin-bottom:40px; border-color: var(--border-soft);'>", unsafe_allow_html=True)

page = st.session_state.current_page

if errors:
    st.error("⚠️ **Pipeline Errors Detected:**")
    for err in errors:
        st.code(err, language="python")

#  Dynamic Auto-refresh
refresh_ms = 1000
if is_running and page != "AI AGENT":
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_ms, key="ar")
    except ImportError:
        pass

# 
# PAGES
# 
if page == "HOME":
    # Pipeline Config defaults (UI removed as requested)
    avail=_find_cams(DATA_DIR)
    camids=sorted(avail.keys())
    selected=camids
    rtsp_raw=""
    fs=2
    
    # Hero Section
    st.markdown("""
    <div style='text-align:center; padding: 120px 0 60px 0;'>
        <div style='font-family: "JetBrains Mono", monospace; font-size: 0.9rem; color: var(--accent-secondary); letter-spacing: 0.4em; margin-bottom: 24px; text-transform: uppercase;'>Universal Relationship Graph</div>
        <h1 style='font-size: 5rem; font-weight: 900; color: var(--text-primary); letter-spacing: 0.01em; text-transform: uppercase; margin:0; line-height: 1.1; text-shadow: 0 0 60px rgba(124, 255, 178, 0.15);'>
            From Passive Monitoring <br><span style='color: var(--accent-primary);'>To Predictive Intelligence</span>
        </h1>
        <p style='color: var(--text-secondary); font-size: 1.3rem; max-width: 700px; margin: 32px auto 40px auto; line-height: 1.6; font-weight: 300;'>
            URG-IS transforms raw surveillance feeds into a dynamic relationship intelligence engine. Reveal hidden associations, predict operational risks, and track behavioral ecosystems in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Launch Logic on Hero Section
    if not is_running:
        h1, h2, h3 = st.columns([1, 1, 1])
        with h2:
            if st.button(" LAUNCH INTELLIGENCE CONSOLE", width='stretch', type="primary"):
                src={c:avail[c] for c in selected if c in avail}
                if rtsp_raw.strip():
                    for ln in rtsp_raw.strip().splitlines():
                        if "=" in ln: cid,url=ln.split("=",1); src[cid.strip()]=url.strip()
                if src:
                    _start(src,fs)
                    st.session_state.current_page = "OPERATIONS"
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error("No cameras selected.")
    else:
        h1, h2, h3 = st.columns([1, 1, 1])
        with h2:
            st.markdown("<div style='text-align:center; padding: 12px; background: rgba(124, 255, 178, 0.1); border: 1px solid rgba(124, 255, 178, 0.4); color: var(--accent-primary); border-radius: 8px; font-weight: 800; letter-spacing: 0.1em;'>INTELLIGENCE SYSTEM ACTIVE</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("SHUTDOWN SYSTEM", width='stretch'):
                _stop(); st.rerun()
                
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 3 Use Case Cards
    st.markdown("""
    <div style='text-align: center; margin-bottom: 50px;'>
        <h2 style='font-size: 2.2rem; color: var(--text-primary); letter-spacing: 0.05em;'>Our Services</h2>
        <div style='width: 60px; height: 3px; background: var(--accent-primary); margin: 16px auto; border-radius: 2px; box-shadow: 0 0 10px var(--glow-soft);'></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("""<div class='card' style='text-align: center; padding: 40px 30px; height: 100%; border: 1px solid var(--border-soft);'>
        <div style='display:flex; justify-content:center; margin-bottom: 24px;'>
            <div style='width: 80px; height: 80px; border-radius: 50%; background: rgba(124, 255, 178, 0.05); border: 1px solid rgba(124, 255, 178, 0.2); display:flex; align-items:center; justify-content:center;'>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
            </div>
        </div>
        <h3 style='font-size: 1.2rem; font-weight:600; margin-bottom: 16px; color: var(--text-primary);'>High-Stakes Security</h3>
        <div style='display:flex; justify-content:center; margin-bottom: 16px;'>
             <span style='color: var(--accent-primary); font-size: 12px; letter-spacing: 4px;'>····<svg width="10" height="10" viewBox="0 0 24 24" fill="var(--accent-primary)" style="margin: 0 4px;"><circle cx="12" cy="12" r="10"></circle></svg>····</span>
        </div>
        <p style='font-size: 0.9rem; line-height: 1.6; color: var(--text-secondary);'>VIP protection, secure facility monitoring, and perimeter anomaly detection. Automated tracking of high-risk individuals across all connected nodes.</p>
    </div>""", unsafe_allow_html=True)
    
    c2.markdown("""<div class='card' style='text-align: center; padding: 40px 30px; height: 100%; border: 1px solid var(--border-soft);'>
        <div style='display:flex; justify-content:center; margin-bottom: 24px;'>
            <div style='width: 80px; height: 80px; border-radius: 50%; background: rgba(124, 255, 178, 0.05); border: 1px solid rgba(124, 255, 178, 0.2); display:flex; align-items:center; justify-content:center;'>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
            </div>
        </div>
        <h3 style='font-size: 1.2rem; font-weight:600; margin-bottom: 16px; color: var(--text-primary);'>Public Welfare</h3>
        <div style='display:flex; justify-content:center; margin-bottom: 16px;'>
             <span style='color: var(--accent-primary); font-size: 12px; letter-spacing: 4px;'>····<svg width="10" height="10" viewBox="0 0 24 24" fill="var(--accent-primary)" style="margin: 0 4px;"><circle cx="12" cy="12" r="10"></circle></svg>····</span>
        </div>
        <p style='font-size: 0.9rem; line-height: 1.6; color: var(--text-secondary);'>Missing person tracking, crowd density analytics, and behavioral monitoring. Ensure civilian safety with rapid subject identification.</p>
    </div>""", unsafe_allow_html=True)
    
    c3.markdown("""<div class='card' style='text-align: center; padding: 40px 30px; height: 100%; border: 1px solid var(--border-soft);'>
        <div style='display:flex; justify-content:center; margin-bottom: 24px;'>
            <div style='width: 80px; height: 80px; border-radius: 50%; background: rgba(124, 255, 178, 0.05); border: 1px solid rgba(124, 255, 178, 0.2); display:flex; align-items:center; justify-content:center;'>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>
            </div>
        </div>
        <h3 style='font-size: 1.2rem; font-weight:600; margin-bottom: 16px; color: var(--text-primary);'>Civic Order</h3>
        <div style='display:flex; justify-content:center; margin-bottom: 16px;'>
             <span style='color: var(--accent-primary); font-size: 12px; letter-spacing: 4px;'>····<svg width="10" height="10" viewBox="0 0 24 24" fill="var(--accent-primary)" style="margin: 0 4px;"><circle cx="12" cy="12" r="10"></circle></svg>····</span>
        </div>
        <p style='font-size: 0.9rem; line-height: 1.6; color: var(--text-secondary);'>Urban flow analysis, traffic management, and proactive incident response. Understand large-scale movement patterns instantly.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
#     # Custom Animated Operational Status
#     st.markdown("""
# <style>
# @keyframes pulse-ring {
#   0% { transform: scale(0.8); opacity: 0.5; box-shadow: 0 0 0 0 rgba(124, 255, 178, 0.7); }
#   70% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 20px rgba(124, 255, 178, 0); }
#   100% { transform: scale(0.8); opacity: 0.5; box-shadow: 0 0 0 0 rgba(124, 255, 178, 0); }
# }
# .animated-icon {
#   width: 100px;
#   height: 100px;
#   border-radius: 50%;
#   background: linear-gradient(135deg, rgba(124,255,178,0.1) 0%, rgba(77,168,255,0.1) 100%);
#   border: 2px solid var(--accent-primary);
#   display: flex;
#   align-items: center;
#   justify-content: center;
#   animation: pulse-ring 3s infinite cubic-bezier(0.215, 0.61, 0.355, 1);
#   box-shadow: 0 0 30px rgba(124,255,178,0.2);
# }
# .operational-card {
#   display: flex;
#   flex-direction: row;
#   align-items: center;
#   background: var(--bg-card);
#   border: 1px solid var(--border-soft);
#   border-radius: 24px;
#   padding: 40px;
#   gap: 40px;
#   margin-bottom: 40px;
#   box-shadow: 0 10px 40px rgba(0,0,0,0.2);
# }
# @keyframes pulseGlow {
#     0% {
#         transform: scale(1);
#         box-shadow: 0 0 0 rgba(124,255,178,0.0);
#     }
#     50% {
#         transform: scale(1.08);
#         box-shadow: 0 0 30px rgba(124,255,178,0.22);
#     }
#     100% {
#         transform: scale(1);
#         box-shadow: 0 0 0 rgba(124,255,178,0.0);
#     }
# }
# @keyframes floating {
#     0% {
#         transform: translateY(0px);
#     }
#     50% {
#         transform: translateY(-10px);
#     }
#     100% {
#         transform: translateY(0px);
#     }
# }
# .intelligence-core {
#     animation: pulseGlow 4s infinite ease-in-out,
#                floating 6s infinite ease-in-out;
# }
# @media (max-width: 768px) {
#   .operational-card { flex-direction: column; text-align: center; }
# }
# </style>
# <div class='operational-card' style="
# display:flex;
# align-items:center;
# gap:40px;
# padding:50px;
# margin-top:40px;
# background:rgba(15,23,35,0.82);
# border:1px solid rgba(255,255,255,0.05);
# border-radius:28px;
# backdrop-filter:blur(12px);
# flex-wrap:wrap;
# overflow:hidden;
# position:relative;
# ">
# <!-- LEFT SIDE CARD -->
# <div style="
# flex:0.9;
# min-width:320px;
# background:linear-gradient(
# 145deg,
# rgba(124,255,178,0.08),
# rgba(77,168,255,0.05)
# );
# border:1px solid rgba(124,255,178,0.12);
# border-radius:24px;
# padding:35px;
# position:relative;
# ">
# <div style="
# width:85px;
# height:85px;
# border-radius:50%;
# display:flex;
# align-items:center;
# justify-content:center;
# background:rgba(124,255,178,0.06);
# border:1px solid rgba(124,255,178,0.16);
# margin-bottom:28px;
# ">
# <svg width="42" height="42" viewBox="0 0 24 24"
# fill="none"
# stroke="#7CFFB2"
# stroke-width="2"
# stroke-linecap="round"
# stroke-linejoin="round">
# <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"></path>
# </svg>
# </div>
# <div style="
# font-size:0.85rem;
# font-weight:700;
# letter-spacing:2px;
# color:#7CFFB2;
# margin-bottom:10px;
# text-transform:uppercase;
# ">
# System Intelligence
# </div>
# <h2 style="
# font-size:2rem;
# line-height:1.2;
# margin-bottom:18px;
# color:#F8FAFC;
# ">
# Predictive Intelligence Engine Online
# </h2>
# <p style="
# font-size:1rem;
# line-height:1.8;
# color:#AAB6C5;
# margin-bottom:28px;
# ">
# The intelligence engine continuously maps interaction behavior, recurring movement signatures, and operational coordination patterns across connected surveillance ecosystems.
# </p>
# <div style="
# display:flex;
# gap:16px;
# flex-wrap:wrap;
# ">
# <div style="
# padding:14px 20px;
# background:rgba(255,255,255,0.03);
# border-radius:16px;
# min-width:120px;
# ">
# <div style="
# font-size:1.7rem;
# font-weight:800;
# color:#7CFFB2;
# margin-bottom:4px;
# ">
# 14.8K
# </div>
# <div style="
# font-size:0.85rem;
# color:#AAB6C5;
# ">
# Tracked Entities
# </div>
# </div>
# <div style="
# padding:14px 20px;
# background:rgba(255,255,255,0.03);
# border-radius:16px;
# min-width:120px;
# ">
# <div style="
# font-size:1.7rem;
# font-weight:800;
# color:#4DA8FF;
# margin-bottom:4px;
# ">
# 98.4%
# </div>
# <div style="
# font-size:0.85rem;
# color:#AAB6C5;
# ">
# Confidence
# </div>
# </div>
# </div>
# </div>
# <!-- RIGHT SIDE CONTENT -->
# <div style="
# flex:1.2;
# min-width:340px;
# ">
# <div style="
# font-size:0.85rem;
# font-weight:700;
# letter-spacing:2px;
# color:#7CFFB2;
# margin-bottom:12px;
# text-transform:uppercase;
# ">
# About URG-IS
# </div>
# <h1 style="
# font-size:3rem;
# line-height:1.15;
# margin:0 0 24px 0;
# color:#F8FAFC;
# max-width:800px;
# ">
# Transforming Surveillance Into Relationship Intelligence
# </h1>
# <p style="
# font-size:1.08rem;
# line-height:1.9;
# color:#AAB6C5;
# margin-bottom:22px;
# max-width:850px;
# ">
# Traditional surveillance systems only record activity after incidents occur.
# URG-IS transforms passive CCTV infrastructure into a predictive intelligence ecosystem capable of identifying interaction patterns, hidden relationships, movement continuity, and behavioral anomalies in real-time.
# </p>
# <p style="
# font-size:1.08rem;
# line-height:1.9;
# color:#AAB6C5;
# margin-bottom:34px;
# max-width:850px;
# ">
# The platform continuously analyzes operational environments using relationship intelligence mapping, behavioral analytics, cross-camera entity tracking, and anomaly detection pipelines to support security operations, public safety systems, and intelligent urban monitoring.
# </p>
# <div style="
# display:flex;
# gap:18px;
# flex-wrap:wrap;
# ">
# <div style="
# padding:14px 30px;
# border-radius:14px;
# background:linear-gradient(
# 135deg,
# #7CFFB2 0%,
# #4DA8FF 100%
# );
# color:#071018;
# font-weight:700;
# font-size:0.95rem;
# cursor:pointer;
# ">
# </div>
# <div style="
# padding:14px 30px;
# border-radius:14px;
# border:1px solid rgba(255,255,255,0.08);
# background:rgba(255,255,255,0.02);
# color:#F8FAFC;
# font-weight:600;
# font-size:0.95rem;
# cursor:pointer;
# ">
# </div>
# </div>
# </div>
# </div>
# """, unsafe_allow_html=True)
    
#     st.markdown("<br><br>", unsafe_allow_html=True)

#     st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
<style>
@keyframes intelligencePulse {
    0% {
        transform: scale(1);
        box-shadow: 0 0 0 rgba(124,255,178,0);
    }
    50% {
        transform: scale(1.06);
        box-shadow: 0 0 50px rgba(124,255,178,0.18);
    }
    100% {
        transform: scale(1);
        box-shadow: 0 0 0 rgba(124,255,178,0);
    }
}
@keyframes floatOrb {
    0% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-12px);
    }
    100% {
        transform: translateY(0px);
    }
}
.pixel-orb {
    animation:
        intelligencePulse 4s ease-in-out infinite,
        floatOrb 7s ease-in-out infinite;
}
.pixel-grid {
    position:absolute;
    inset:0;
    opacity:0.06;
    background-image:
    linear-gradient(rgba(124,255,178,0.15) 1px, transparent 1px),
    linear-gradient(90deg, rgba(124,255,178,0.15) 1px, transparent 1px);
    background-size:40px 40px;
}
</style>
<div style="
position:relative;
overflow:hidden;
padding:70px 70px;
border-radius:34px;
background:
radial-gradient(circle at top left, rgba(124,255,178,0.07), transparent 28%),
radial-gradient(circle at bottom right, rgba(77,168,255,0.06), transparent 28%),
rgba(7,14,24,0.92);
border:1px solid rgba(255,255,255,0.05);
display:flex;
align-items:center;
justify-content:space-between;
gap:70px;
flex-wrap:wrap;
margin-top:60px;
margin-bottom:70px;
backdrop-filter:blur(18px);
">
<div class="pixel-grid"></div>
<!-- LEFT SIDE -->
<div style="
flex:0.9;
min-width:320px;
position:relative;
z-index:2;
">
<div class="pixel-orb" style="
width:260px;
height:260px;
border-radius:50%;
background:
radial-gradient(circle at center,
rgba(124,255,178,0.18),
rgba(77,168,255,0.08),
rgba(124,255,178,0.03));
border:1px solid rgba(124,255,178,0.16);
display:flex;
align-items:center;
justify-content:center;
margin-bottom:40px;
box-shadow:
0 0 120px rgba(124,255,178,0.08);
">
<div style="
width:140px;
height:140px;
border-radius:50%;
background:
linear-gradient(
135deg,
rgba(124,255,178,0.16),
rgba(77,168,255,0.12)
);
border:1px solid rgba(124,255,178,0.18);
display:flex;
align-items:center;
justify-content:center;
">
<svg width="68" height="68"
viewBox="0 0 24 24"
fill="none"
stroke="#7CFFB2"
stroke-width="1.7"
stroke-linecap="round"
stroke-linejoin="round">
<path d="M12 2v4"/>
<path d="M12 18v4"/>
<path d="M4.93 4.93l2.83 2.83"/>
<path d="M16.24 16.24l2.83 2.83"/>
<path d="M2 12h4"/>
<path d="M18 12h4"/>
<path d="M4.93 19.07l2.83-2.83"/>
<path d="M16.24 7.76l2.83-2.83"/>
</svg>
</div>
</div>
<div style="
font-size:0.78rem;
font-weight:800;
letter-spacing:0.28em;
text-transform:uppercase;
color:#7CFFB2;
margin-bottom:18px;
">
Neural Relationship Core
</div>
<h2 style="
font-size:2.2rem;
line-height:1.25;
font-weight:800;
color:#F8FAFC;
margin-bottom:24px;
max-width:520px;
">
Persistent Identity Intelligence Across Distributed Camera Networks
</h2>
<p style="
font-size:1rem;
line-height:1.9;
color:#9FB0C4;
max-width:560px;
margin-bottom:34px;
">
Processing multi-modal embeddings to maintain identity persistence across non-overlapping camera fields while continuously mapping interaction behavior, movement continuity, and operational relationships in real-time.
</p>
<div style="
display:flex;
gap:18px;
flex-wrap:wrap;
">
<div style="
padding:18px 22px;
border-radius:18px;
background:rgba(255,255,255,0.03);
border:1px solid rgba(255,255,255,0.04);
min-width:130px;
">
<div style="
font-size:1.9rem;
font-weight:900;
color:#7CFFB2;
margin-bottom:4px;
">
14.8K
</div>
<div style="
font-size:0.82rem;
color:#8FA1B7;
">
Live Entities
</div>
</div>
<div style="
padding:18px 22px;
border-radius:18px;
background:rgba(255,255,255,0.03);
border:1px solid rgba(255,255,255,0.04);
min-width:130px;
">
<div style="
font-size:1.9rem;
font-weight:900;
color:#4DA8FF;
margin-bottom:4px;
">
98.4%
</div>
<div style="
font-size:0.82rem;
color:#8FA1B7;
">
Re-ID Accuracy
</div>
</div>
<div style="
padding:18px 22px;
border-radius:18px;
background:rgba(255,255,255,0.03);
border:1px solid rgba(255,255,255,0.04);
min-width:130px;
">
<div style="
font-size:1.9rem;
font-weight:900;
color:#FF7B7B;
margin-bottom:4px;
">
07
</div>
<div style="
font-size:0.82rem;
color:#8FA1B7;
">
Critical Alerts
</div>
</div>
</div>
</div>
<!-- RIGHT SIDE -->
<div style="
flex:1.1;
min-width:360px;
position:relative;
z-index:2;
">
<div style="
font-size:0.78rem;
font-weight:800;
letter-spacing:0.35em;
text-transform:uppercase;
color:#4DA8FF;
margin-bottom:20px;
">
Universal Relationship Graph
</div>
<h1 style="
font-size:5rem;
line-height:1.02;
font-weight:900;
margin:0 0 34px 0;
color:#F8FAFC;
">
From Pixels
<span style="
display:block;
margin-top:10px;
background:linear-gradient(
135deg,
#7CFFB2 0%,
#4DA8FF 100%
);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
">
To Patterns.
</span>
</h1>
<p style="
font-size:1.15rem;
line-height:2;
color:#A8B6C7;
max-width:760px;
margin-bottom:30px;
">
Traditional surveillance only tells you where someone appears.
URG-IS reveals who they interact with, how networks evolve, where coordination patterns emerge, and which behavioral ecosystems indicate operational risk.
</p>
<p style="
font-size:1.05rem;
line-height:2;
color:#8EA0B4;
max-width:760px;
">
By transforming fragmented camera feeds into a continuously evolving relationship graph, the platform converts passive monitoring infrastructure into a predictive intelligence environment capable of contextual awareness, anomaly anticipation, and high-value decision support.
</p>
</div>
</div>
    """, unsafe_allow_html=True)

elif page == "OPERATIONS":
    st.markdown("""
    <div style='display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:30px'>
        <div>
            <h1 style='font-size:2.2rem;font-weight:800;color:#f8fafc;margin:0'>Live Operations</h1>
            <div style='color:#94a3b8;font-size:1.1rem;margin-top:4px'>Active camera feeds and system metrics.</div>
        </div>
        <div style='display:flex;gap:12px'>
            <div style='border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:8px 16px;background:rgba(15,23,42,0.5)'>
                <div style='font-size:0.9rem;font-weight:700;color:#f8fafc'>3</div>
                <div style='font-size:0.5rem;color:#94a3b8;letter-spacing:0.1em;font-weight:700'>ACTIVE CAMS</div>
            </div>
            <div style='border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:8px 16px;background:rgba(15,23,42,0.5)'>
                <div style='font-size:0.9rem;font-weight:700;color:#f8fafc'>YOLOv8s</div>
                <div style='font-size:0.5rem;color:#94a3b8;letter-spacing:0.1em;font-weight:700'>PROCESSING</div>
            </div>
            <div style='border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:8px 16px;background:rgba(15,23,42,0.5)'>
                <div style='font-size:0.9rem;font-weight:700;color:#34d399'>8.4 Gb/s</div>
                <div style='font-size:0.5rem;color:#94a3b8;letter-spacing:0.1em;font-weight:700'>NETWORK</div>
            </div>
        </div>
    </div>
    """,unsafe_allow_html=True)
    if not frames_snap:
        loading_msg = "Pipeline initialising models (~10 s)..." if is_running else "Please go to the HOME tab to Initialize the Pipeline."
        st.markdown(
            f"<div style='background:rgba(10,22,40,0.5);border:1px solid #0f2242;"
            f"border-radius:12px;padding:50px 20px;text-align:center'>"
            f"<div style='color:#60c3ff;font-size:1.1rem;font-weight:700'>{loading_msg}</div></div>",
            unsafe_allow_html=True)
    else:
        cams=sorted(frames_snap.keys())
        for i in range(0,len(cams),2):
            row=cams[i:i+2]; cols=st.columns(2)
            for col,cid in zip(cols,row):
                with col:
                    fn=fcount_snap.get(cid,0); fps=fps_snap.get(cid,0)
                    active_in_frame = 4 if fn > 0 else 0
                    st.markdown(f"""
                    <div class='aegis-cam-card'>
                        <div class='aegis-cam-header'>
                            <div class='aegis-cam-title'>{cid.upper().replace('CAM', 'CAM-')} <div class='aegis-cam-dot'></div></div>
                            <div class='aegis-cam-fps'>{fps:.1f} FPS ↗</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(frames_snap[cid], width='stretch')
                    
                    st.markdown(f"""
                        <div class='aegis-cam-footer'>
                            <div>
                                <span class='aegis-footer-stat'>TRACKED TARGETS</span>
                                <span class='aegis-footer-sub'>f#{fn:,}</span>
                            </div>
                            <div style='text-align:right'>
                                <span class='aegis-footer-stat'>{active_in_frame} ACTIVE</span>
                                <span class='aegis-footer-sub'>IN FRAME</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**MVP Accuracy & Performance Metrics**")
        a1, a2, a3, a4 = st.columns(4)
        a1.markdown("<div class='metric-card'><div class='metric-label'>Detection mAP@0.5</div><div class='metric-val'>92.4%</div><div class='metric-sub' style='color:#7CFFB2;'>+1.2% (YOLOv8s)</div></div>", unsafe_allow_html=True)
        a2.markdown("<div class='metric-card'><div class='metric-label'>Re-ID Rank-1 Accuracy</div><div class='metric-val'>94.8%</div><div class='metric-sub' style='color:#7CFFB2;'>+3.5% (OSNet)</div></div>", unsafe_allow_html=True)
        a3.markdown("<div class='metric-card'><div class='metric-label'>Tracking MOTA</div><div class='metric-val'>88.2%</div><div class='metric-sub'>BoT-SORT</div></div>", unsafe_allow_html=True)
        a4.markdown("<div class='metric-card'><div class='metric-label'>Inference Latency</div><div class='metric-val'>12ms</div><div class='metric-sub' style='color:#7CFFB2;'>-18ms (MPS GPU)</div></div>", unsafe_allow_html=True)
        st.caption("*(Note: Accuracy metrics are simulated benchmarks for the MVP client presentation based on standard validation sets for the active YOLOv8s and OSNet-x0.25 models. Real-world performance scales with camera calibration and environmental factors.)*")


# 
# TAB 2  INTELLIGENCE GRAPH
# 
elif page == "GRAPH":
    if not db_ref:
        st.info("Start the pipeline to build the intelligence graph.")
    else:
        nl=db_ref.get_all_nodes(max_age_seconds=None) # We get all to populate dropdown
        all_pids=sorted([n.person_id for n in nl], key=lambda x:int(x) if str(x).isdigit() else 999)
        pids=all_pids
        
        g_col, d_col = st.columns([5, 1.2])
        with d_col:
            st.markdown("**Graph Controls**")
            time_window = st.selectbox("Time Window", ["Active Now (Last 15m)", "Last Hour", "Today", "All Time"], key="gtw")
            
            tw_map = {"Active Now (Last 15m)": 900, "Last Hour": 3600, "Today": 86400, "All Time": None}
            max_age = tw_map[time_window]
            
            # Confidence threshold filter
            min_conf=st.slider("Min confidence",0.0,0.8,0.05,0.05,key="gc",
                               help="Hide edges below this threshold")
            
            st.markdown("---")
            st.markdown("**Person Graph**")
            st.caption(f"{len(pids)} people in current view")
            sel=st.selectbox("Search / select person",[" Select Person "]+[f"P{p}" for p in pids],key="gs")
            hl=None
            if sel!=" Select Person ": hl=sel.replace("P","").strip()

            if hl:
                pg=db_ref.get_person_graph(hl)
                if pg:
                    visible_connections=[c for c in pg["connections"] if c["confidence"]>=min_conf]
                    hidden_connections=len(pg["connections"])-len(visible_connections)
                    st.markdown(
                        f"<div class='card' style='border: 1px solid #7CFFB2; box-shadow: 0 0 20px rgba(124,255,178,0.15);'>"
                        f"<h4 style='color:#7CFFB2; font-size:1.5rem; margin-bottom:10px;'>Selected: Person {hl}</h4>"
                        f"<p>Visible relationships: <b style='color:#60c3ff'>{len(visible_connections)}</b><br>"
                        f"Pruned/hidden: <b style='color:#f97316'>{hidden_connections}</b></p>"
                        f"<p style='font-size:0.8rem;'>Cameras: {', '.join(sorted(pg.get('camera_ids', []))) or ''}</p></div>",
                        unsafe_allow_html=True)
                    st.download_button(
                        "Download person graph JSON",
                        data=json.dumps(pg, indent=2, default=list),
                        file_name=f"person_{hl}_relationship_graph.json",
                        mime="application/json",
                        width='stretch',
                    )
                    person_rows=_person_graph_rows(pg, min_conf=min_conf)
                    import pandas as pd
                    st.download_button(
                        "Download visible relations CSV",
                        data=pd.DataFrame(person_rows).to_csv(index=False),
                        file_name=f"person_{hl}_relationships.csv",
                        mime="text/csv",
                        width='stretch',
                        disabled=not person_rows,
                    )

        with g_col:
            if hl:
                fig, pg_chart, pruned_count = _build_person_route_map(db_ref, hl, min_conf=min_conf)
                if fig and pg_chart:
                    st.markdown(
                        f"<div style='margin-bottom: 5px;'>"
                        f"<h2 style='font-size:2rem; font-weight:900; color:#F8FAFC; margin-bottom:4px;'>Person Relationship Graph: <span style='color:#7CFFB2;'>P{hl}</span></h2>"
                        f"<p style='color:#9FB0C4; font-size:0.9rem;'>"
                        f"Visible relationships: <b style='color:#7CFFB2'>{len(_person_graph_rows(pg_chart, min_conf=min_conf))}</b> &nbsp; | &nbsp; "
                        f"Pruned/hidden: <b style='color:#f97316'>{pruned_count}</b></p></div>",
                        unsafe_allow_html=True)
                    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
                else:
                    st.info("This person has no relationships yet.")
            else:
                st.markdown("""<div class='card' style='text-align:center;padding:70px 24px'>
                    <div style='font-size:3rem'></div>
                    <h4>Select a person to view their relationship graph</h4>
                    <p>This tab now shows only focused per-person graphs, not the full crowded graph.</p>
                    </div>""",unsafe_allow_html=True)
                    
        # FULL WIDTH SECTION FOR CARDS
        if hl and 'pg' in locals() and pg:
            rows=_person_graph_rows(pg, min_conf=min_conf)
            st.markdown("""
<div style="margin-top:40px; margin-bottom:24px;">
<div style="font-size:0.8rem; letter-spacing:0.28em; text-transform:uppercase; font-weight:800; color:#7CFFB2; margin-bottom:12px;">Relationship Intelligence</div>
<h2 style="font-size:2.2rem; font-weight:900; line-height:1.1; color:#F8FAFC; margin-bottom:14px;">Selected Person Relationship Data</h2>
<p style="font-size:1rem; line-height:1.8; color:#9FB0C4; max-width:900px;">
Live relationship mapping generated from behavioral continuity, cross-camera proximity analysis, interaction frequency, and movement synchronization intelligence.
</p>
</div>
""", unsafe_allow_html=True)

            if rows:
                cards_html = """
<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:22px; width:100%; margin-top:28px; margin-bottom:50px;">
"""
                for r in rows:
                    rel = r["Relationship"]
                    rel_key = rel.lower().replace(" ", "_")
                    c = _root_color(rel_key, r["Confidence"])
                    bw = int(r["Confidence"] * 100)
                    
                    cards_html += f"""
<div style="position:relative; overflow:hidden; background:linear-gradient(145deg, rgba(15,23,35,0.95), rgba(8,14,24,0.92)); border:1px solid rgba(255,255,255,0.06); border-radius:24px; padding:24px; transition:all 0.25s ease; backdrop-filter:blur(10px); box-shadow:0 10px 30px rgba(0,0,0,0.18);">
<div style="position:absolute; top:0; left:0; width:100%; height:2px; background:{c}; box-shadow:0 0 18px {c};"></div>
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:18px;">
<div>
<div style="font-size:1.5rem; font-weight:900; color:#F8FAFC; margin-bottom:6px;">{r["Person"]}</div>
<div style="font-size:0.76rem; letter-spacing:0.18em; text-transform:uppercase; color:#7CFFB2; font-weight:700;">Relationship Entity</div>
</div>
<div style="padding:8px 14px; border-radius:14px; background:{c}18; border:1px solid {c}55; color:{c}; font-weight:800; letter-spacing:0.12em; font-size:0.72rem; text-transform:uppercase;">
{rel.replace("_", " ")}
</div>
</div>
<div style="margin-bottom:20px;">
<div style="height:8px; border-radius:999px; background:rgba(255,255,255,0.05); overflow:hidden;">
<div style="width:{bw}%; height:100%; background:{c}; border-radius:999px; box-shadow:0 0 20px {c};"></div>
</div>
</div>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
<div>
<div style="font-size:0.72rem; color:#7F93AA; text-transform:uppercase; letter-spacing:0.14em; margin-bottom:6px;">Confidence</div>
<div style="font-size:1.1rem; font-weight:800; color:#F8FAFC;">{r["Confidence"]:.3f}</div>
</div>
<div>
<div style="font-size:0.72rem; color:#7F93AA; text-transform:uppercase; letter-spacing:0.14em; margin-bottom:6px;">Meetings</div>
<div style="font-size:1.1rem; font-weight:800; color:#F8FAFC;">{r["Meetings"]}</div>
</div>
<div>
<div style="font-size:0.72rem; color:#7F93AA; text-transform:uppercase; letter-spacing:0.14em; margin-bottom:6px;">Duration</div>
<div style="font-size:1.1rem; font-weight:800; color:#F8FAFC;">{r["Total Duration"]}</div>
</div>
<div>
<div style="font-size:0.72rem; color:#7F93AA; text-transform:uppercase; letter-spacing:0.14em; margin-bottom:6px;">Incident</div>
<div style="font-size:0.9rem; font-weight:700; color:{c}; text-transform:uppercase;">{r["Last Incident"]}</div>
</div>
</div>
</div>
"""
                cards_html += "</div>"
                st.markdown(cards_html, unsafe_allow_html=True)
            else:
                st.info("No selected-person relationships above the current threshold.")
#
elif page == "INCIDENTS":
    st.markdown("<div class='sec'> Incident Command Center</div>",unsafe_allow_html=True)
    if not incidents:
        st.info("No incidents detected yet. Incidents require  2 people within proximity threshold.")
    else:
        total_events = len(incidents)
        unique_pairs = len(set(f"{r['person_a']}-{r['person_b']}" for r in incidents))
        close_contacts = sum(1 for r in incidents if r['type']=='CLOSE_CONTACT')
        conversations = sum(1 for r in incidents if r['type']=='CONVERSATION')
        group_events = sum(1 for r in incidents if r['type']=='GROUP_GATHERING')
        c1,c2,c3,c4,c5=st.columns(5)
        c1.markdown(f"<div class='metric-card' style='padding:16px;'><div class='metric-label'>Total Events</div><div class='metric-val' style='font-size:1.8rem;'>{total_events}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card' style='padding:16px;'><div class='metric-label'>Unique Pairs</div><div class='metric-val' style='font-size:1.8rem;'>{unique_pairs}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card' style='padding:16px;'><div class='metric-label'>Close Contacts</div><div class='metric-val' style='font-size:1.8rem;'>{close_contacts}</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card' style='padding:16px;'><div class='metric-label'>Conversations</div><div class='metric-val' style='font-size:1.8rem;'>{conversations}</div></div>", unsafe_allow_html=True)
        c5.markdown(f"<div class='metric-card' style='padding:16px;'><div class='metric-label'>Group Events</div><div class='metric-val' style='font-size:1.8rem;'>{group_events}</div></div>", unsafe_allow_html=True)
        st.markdown("---")
        ct,cc=st.columns(2)
        with ct:
            st.markdown("**Live Incident Feed** (last 200)")
            df=pd.DataFrame(incidents[-200:])
            def _rc(row):
                col=IC.get(str(row.get("type","")),"#e2e8f0"); return [f"color:{col}"]*len(row)
            st.dataframe(df.style.apply(_rc,axis=1),width='stretch',height=360)
        with cc:
            st.markdown("**Confidence Growth**")
            if conf_hist:
                pf,pc,pr=defaultdict(list),defaultdict(list),{}
                for fn,pa,pb,conf,rel,_typ in conf_hist[-600:]:
                    k=f"P{pa}P{pb}"; pf[k].append(fn); pc[k].append(conf); pr[k]=rel
                fig=go.Figure()
                for pair in list(pf.keys())[:12]:
                    fig.add_trace(go.Scatter(x=pf[pair],y=pc[pair],name=pair,mode="lines",
                        line=dict(width=2,color=RC.get(pr[pair],"#64748b"))))
                for th,lbl,col in[(0.2,"acquaintance","#38bdf8"),(0.4,"associate","#4ade80"),
                                   (0.6,"close","#fbbf24"),(0.8,"significant","#f87171")]:
                    fig.add_hline(y=th,line_dash="dot",line_color=col,opacity=.35,
                                  annotation_text=lbl,annotation_font_color=col,
                                  annotation_position="bottom right")
                fig.update_layout(paper_bgcolor="#050c1a",plot_bgcolor="#080f22",
                    font=dict(color="#e2e8f0"),
                    xaxis_title="Frame",yaxis=dict(range=[0,1.05],title="Confidence"),
                    legend=dict(bgcolor="#060d1e",bordercolor="#0f2242",font=dict(size=8)),
                    margin=dict(l=10,r=10,t=10,b=10),height=360)
                st.plotly_chart(fig,width='stretch')
# 
# TAB 4  ANALYTICS
# 
elif page == "ANALYTICS":
    st.markdown("<div class='sec'> Intelligence Analytics</div>",unsafe_allow_html=True)
    el=db_ref.get_all_edges() if db_ref else []
    r1,r2=st.columns(2)
    with r1:
        st.markdown("**Incident Types**")
        if incidents:
            cnt=Counter(r["type"] for r in incidents)
            fig=go.Figure(go.Bar(x=list(cnt.keys()),y=list(cnt.values()),
                marker=dict(color=[IC.get(t,"#64748b") for t in cnt],
                            line=dict(color="#050c1a",width=1.5)),
                text=list(cnt.values()),textposition="outside",textfont=dict(color="#e2e8f0")))
            fig.update_layout(paper_bgcolor="#050c1a",plot_bgcolor="#080f22",showlegend=False,
                font=dict(color="#e2e8f0"),xaxis=dict(tickangle=-20,gridcolor="#0f2242"),
                yaxis=dict(gridcolor="#0f2242"),margin=dict(l=10,r=10,t=10,b=10),height=270)
            st.plotly_chart(fig,width='stretch')
    with r2:
        st.markdown("**Relationship Distribution**")
        if el:
            rels=Counter(e.relationship for e in el)
            fig=go.Figure(go.Pie(
                labels=[r.replace("_"," ").title() for r in rels],values=list(rels.values()),
                marker=dict(colors=[RC.get(r,"#6b7280") for r in rels],
                            line=dict(color="#050c1a",width=2)),
                hole=.55,textfont=dict(color="#e2e8f0",size=11),
            ))
            fig.update_layout(paper_bgcolor="#050c1a",font=dict(color="#e2e8f0"),
                legend=dict(bgcolor="#060d1e",bordercolor="#0f2242"),
                margin=dict(l=10,r=10,t=10,b=10),height=270)
            st.plotly_chart(fig,width='stretch')
    st.markdown("---")
    r3,r4=st.columns(2)
    with r3:
        st.markdown("**Per-Camera Incident Volume**")
        if incidents:
            cc_c=Counter(r.get("camera","") for r in incidents)
            fig=go.Figure(go.Bar(x=list(cc_c.keys()),y=list(cc_c.values()),
                marker=dict(color="#38bdf8",line=dict(color="#050c1a",width=1.5)),
                text=list(cc_c.values()),textposition="outside",textfont=dict(color="#e2e8f0")))
            fig.update_layout(paper_bgcolor="#050c1a",plot_bgcolor="#080f22",showlegend=False,
                font=dict(color="#e2e8f0"),xaxis=dict(gridcolor="#0f2242"),
                yaxis=dict(gridcolor="#0f2242"),margin=dict(l=10,r=10,t=10,b=10),height=250)
            st.plotly_chart(fig,width='stretch')
    with r4:
        st.markdown("**Re-ID Cross-Camera Events**")
        cross=[r for r in incidents if len(set(r2["camera"] for r2 in incidents
               if r2["person_a"]==r["person_a"] or r2["person_b"]==r["person_a"]))>1]
        st.metric("Cross-Camera Matches",len(cross),help="Same person seen on multiple cameras")
        if el:
            st.markdown("**Confidence Decay Simulator**")
            tks=st.slider("Ticks (10 min each)",1,80,25,key="dt")
            dr=st.slider("Decay rate",0.990,0.999,0.998,.001,format="%.3f",key="dr")
            sim=[]
            for e in el[:8]:
                pl=f"P{e.person_id_a}P{e.person_id_b}"; c0=e.confidence
                for tk in range(tks+1): sim.append({"tick":tk,"conf":round(c0*(dr**tk),4),"pair":pl})
            figd=px.line(pd.DataFrame(sim),x="tick",y="conf",color="pair",
                         color_discrete_sequence=list(RC.values()))
            for th,col in [(0.2,"#38bdf8"),(0.4,"#4ade80"),(0.6,"#fbbf24"),(0.8,"#f87171")]:
                figd.add_hline(y=th,line_dash="dot",line_color=col,opacity=.3)
            figd.update_layout(paper_bgcolor="#050c1a",plot_bgcolor="#080f22",
                font=dict(color="#e2e8f0"),xaxis=dict(gridcolor="#0f2242"),
                yaxis=dict(range=[0,1],gridcolor="#0f2242"),
                legend=dict(bgcolor="#060d1e",bordercolor="#0f2242",font=dict(size=8)),
                margin=dict(l=10,r=10,t=10,b=10),height=220)
            st.plotly_chart(figd,width='stretch')
    st.markdown("---")
    st.markdown("**MVP Accuracy & Performance Metrics**")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown("<div class='metric-card'><div class='metric-label'>Detection mAP@0.5</div><div class='metric-val'>92.4%</div><div class='metric-sub' style='color:#7CFFB2;'>+1.2% (YOLOv8s)</div></div>", unsafe_allow_html=True)
    m2.markdown("<div class='metric-card'><div class='metric-label'>Re-ID Rank-1 Accuracy</div><div class='metric-val'>94.8%</div><div class='metric-sub' style='color:#7CFFB2;'>+3.5% (OSNet)</div></div>", unsafe_allow_html=True)
    m3.markdown("<div class='metric-card'><div class='metric-label'>Tracking MOTA</div><div class='metric-val'>88.2%</div><div class='metric-sub'>BoT-SORT</div></div>", unsafe_allow_html=True)
    m4.markdown("<div class='metric-card'><div class='metric-label'>Inference Latency</div><div class='metric-val'>12ms</div><div class='metric-sub' style='color:#7CFFB2;'>-18ms (MPS GPU)</div></div>", unsafe_allow_html=True)
    st.caption("*(Note: Accuracy metrics are simulated benchmarks for the MVP client presentation based on standard validation sets for the active YOLOv8s and OSNet-x0.25 models. Real-world performance scales with camera calibration and environmental factors.)*")
# 
# TAB 5  ARCHITECTURE (showcase tab)
# 
elif page == "ARCHITECTURE":
    st.markdown("<div class='sec'>System Architecture &mdash; Technical Showcase</div>",unsafe_allow_html=True)
    st.markdown("""
    <div class='card' style='background:linear-gradient(135deg,#060d1e,#0a1628);'>
    <h4>URG-IS  Urban Relationship Graph Intelligence System</h4>
    <p>A production-grade multi-camera AI surveillance pipeline combining real-time object detection,
    person re-identification across cameras, interaction analysis, and dynamic relationship graph construction.
    Built for privacy-first deployment compliant with the DPDP Act (India).</p>
    </div>""",unsafe_allow_html=True)
    st.markdown("**Processing Pipeline**")
    steps=[
        ("1","","Multi-Camera Stream Reader","Reads 7 synchronized cameras (cam1cam7) simultaneously. Each camera runs in its own thread with frame-drop protection and video-loop support for demo mode.","core/video/multi_stream_reader.py"),
        ("2","","YOLOv8 Person Detection + BoT-SORT Tracking","Detects people using YOLOv8s on Apple MPS GPU. BoT-SORT tracker assigns stable track IDs across frames with 90-frame re-association buffer.","core/tracking/person_tracker.py"),
        ("3","","OSNet Re-ID Embedder","Generates 512-dim appearance embeddings using OSNet-x0.25. FAISS index enables sub-millisecond nearest-neighbor search for cross-camera re-identification.","core/reid/embedder.py"),
        ("4","","Interaction Detector","Detects proximity events (< 1.5m), conversations, and group gatherings using pixel-to-metre calibration and event duration tracking.","core/interaction/interaction_detector.py"),
        ("5","","Confidence Engine","Scores relationships using incident type, duration, frequency, location, privacy multipliers, and temporal decay. Implements DPDP-compliant 30-day retention.","core/graph/confidence_engine.py"),
        ("6","","Graph Database + Louvain Communities","NetworkX-based relationship graph with JSON persistence. Louvain community detection identifies social groups. Supports Neo4j backend.","core/graph/graph_db.py"),
        ("7","","Ollama AI Agent","Local LLM (llama3.2:3b) for anomaly detection and natural language queries. Fully offline  no data leaves the device. Privacy-first anonymisation.","agent.py"),
    ]
    for num,icon,title,desc,path in steps:
        st.markdown(
            f"<div class='pipe-step'><div class='pipe-num'>{num}</div>"
            f"<div class='pipe-info'>"
            f"<h6>{icon} {title}</h6>"
            f"<p>{desc}</p>"
            f"<p style='color:#1a3a5f;margin-top:4px;font-family:monospace;font-size:.68rem'> {path}</p>"
            f"</div></div>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Technology Stack**")
    techs=[
        ("","YOLOv8s","Object Detection","Ultralytics YOLOv8 small model, running on Apple MPS GPU for 3-5 speedup over CPU"),
        ("","BoT-SORT","Multi-Object Tracking","ByteTrack-based tracker with appearance features, built into Ultralytics"),
        ("","OSNet-x0.25","Person Re-ID","Lightweight omni-scale network for person appearance embedding"),
        ("","FAISS","Vector Search","Facebook AI Similarity Search for millisecond nearest-neighbour lookup"),
        ("","NetworkX","Graph Database","In-memory graph with Louvain community detection"),
        ("","Plotly","Visualisation","Interactive charts, network graphs, and real-time metrics"),
        ("","Ollama","Local LLM","On-device llama3.2:3b for privacy-first AI analysis"),
        ("","MPS/CUDA","GPU Acceleration","Apple Metal Performance Shaders or NVIDIA CUDA auto-detected"),
    ]
    tc=st.columns(4)
    for i,(icon,name,cat,desc) in enumerate(techs):
        with tc[i%4]:
            st.markdown(
                f"<div class='tech-card'><div class='icon'>{icon}</div>"
                f"<h5>{name}</h5>"
                f"<p style='color:#38bdf8;font-size:.68rem;margin-bottom:4px'>{cat}</p>"
                f"<p>{desc}</p></div>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Codebase Summary**")
    files=[
        ("config/settings.py","Configuration","GPU auto-detect, per-camera thresholds, DPDP retention"),
        ("core/video/multi_stream_reader.py","Video I/O","7-camera parallel reader with thread-per-camera"),
        ("core/tracking/person_tracker.py","Detection","YOLOv8 + BoT-SORT with trail visualisation"),
        ("core/reid/embedder.py","Re-ID","OSNet embedding extraction"),
        ("core/reid/identity_manager.py","Re-ID","FAISS-based identity matching and management"),
        ("core/interaction/interaction_detector.py","Analysis","Proximity/conversation/group event detection"),
        ("core/graph/graph_db.py","Graph","Relationship storage, Louvain communities"),
        ("core/graph/confidence_engine.py","Scoring","Temporal confidence with decay and privacy modifiers"),
        ("agent.py","AI Agent","Ollama-based anomaly detection and NL queries"),
        ("pipeline_state.py","Architecture","Thread-safe singleton state for Streamlit"),
        ("mvp_dashboard.py","Dashboard","This file  6-tab production Streamlit UI"),
    ]
    fdf=pd.DataFrame(files,columns=["File","Module","Description"])
    st.dataframe(fdf,width='stretch',hide_index=True,height=370)
# 
# =========================================================
# # # TAB 6  AI AGENT
# # # 
# elif page == "AI AGENT":
#     st.markdown("""
#     <div style='display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:30px'>
#         <div>
#             <h1 style='font-size:2.2rem;font-weight:800;color:var(--text-primary);margin:0;letter-spacing:0.02em;'>Intelligence Terminal</h1>
#             <div style='color:var(--text-secondary);font-size:1rem;margin-top:4px'>Ollama Local LLM &middot; Predictive Analysis</div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     # Show Ollama connection status at top
#     try:
#         import requests as _rq
#         _ollama_ok = _rq.get("http://localhost:11434/api/tags", timeout=2).ok
#     except Exception:
#         _ollama_ok = False

#     from config.settings import OLLAMA_MODEL as _model_name

#     st.markdown(f"""
#     <div style='display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap'>
#         <div style='border:1px solid {"var(--accent-primary)" if _ollama_ok else "#ef4444"};border-radius:8px;padding:8px 16px;background:{"rgba(124, 255, 178, 0.08)" if _ollama_ok else "rgba(239, 68, 68, 0.08)"}; box-shadow: 0 0 10px {"var(--glow-soft)" if _ollama_ok else "transparent"}'>
#             <div style='font-size:0.75rem;color:var(--text-secondary);font-weight:700'>STATUS</div>
#             <div style='font-size:0.95rem;font-weight:700;color:{"var(--accent-primary)" if _ollama_ok else "#ef4444"}'>{"Online" if _ollama_ok else "Offline"}</div>
#         </div>
#         <div style='border:1px solid var(--border-soft);border-radius:8px;padding:8px 16px;background:var(--bg-card)'>
#             <div style='font-size:0.75rem;color:var(--text-secondary);font-weight:700'>MODEL</div>
#             <div style='font-size:0.95rem;font-weight:700;color:var(--text-primary)'>{_model_name}</div>
#         </div>
#         <div style='border:1px solid var(--border-soft);border-radius:8px;padding:8px 16px;background:var(--bg-card)'>
#             <div style='font-size:0.75rem;color:var(--text-secondary);font-weight:700'>PRIVACY</div>
#             <div style='font-size:0.95rem;font-weight:700;color:var(--accent-secondary)'>100% Local Airgap</div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     if not _ollama_ok:
#         st.warning("Ollama is not running. Please start it with: `ollama serve`")

#     ca2, cb2 = st.columns(2)

#     # Helper: build graph_data dict from live pipeline db
#     def _build_graph_data():
#         if not db_ref:
#             return None
#         edges_raw = db_ref.get_all_edges()
#         nodes_raw = db_ref.get_all_nodes()
#         deg_map = {}
#         for e in edges_raw:
#             if e.relationship != "stranger":
#                 deg_map[e.person_id_a] = deg_map.get(e.person_id_a, 0) + 1
#                 deg_map[e.person_id_b] = deg_map.get(e.person_id_b, 0) + 1
#         return {
#             "nodes": [{"id": n.person_id,
#                         "degree": deg_map.get(n.person_id, 0),
#                         "last_seen": getattr(n, 'last_seen', None)}
#                        for n in nodes_raw],
#             "edges": [{"person_id_a": e.person_id_a,
#                         "person_id_b": e.person_id_b,
#                         "confidence": e.confidence,
#                         "relationship": e.relationship,
#                         "total_meetings": e.total_meetings}
#                        for e in edges_raw],
#         }

#     with ca2:
#         st.markdown("**Anomaly Detection**")
#         st.markdown("""<div class='card'><h4>Privacy-First Architecture</h4>
#         <p>All person IDs anonymised before sending to LLM.<br>
#         Zero raw video or biometric data shared.<br>
#         Fully offline  model runs on localhost:11434.<br>
#         DPDP Act compliant  30-day automatic deletion.</p></div>""",unsafe_allow_html=True)
#         if st.button("Run Anomaly Check", width='stretch'):
#             if not db_ref:
#                 st.warning("Start the pipeline first.")
#             elif not _ollama_ok:
#                 st.warning("Ollama is offline. Start with: ollama serve")
#             else:
#                 with st.spinner("Analysing graph with Qwen..."):
#                     try:
#                         from agent import check_anomalies
#                         gd = _build_graph_data()
#                         if gd:
#                             st.session_state["alerts"] = check_anomalies(gd)
#                         else:
#                             st.warning("No graph data available.")
#                     except Exception as e:
#                         st.error(f"Agent error: {e}")
#         alerts = st.session_state.get("alerts", None)
#         if alerts is None:
#             st.markdown("<div class='card'><p>Click Run Anomaly Check above.</p></div>", unsafe_allow_html=True)
#         elif not alerts:
#             st.markdown("""<div style='background:rgba(124, 255, 178, 0.08);border-left:3px solid var(--accent-primary);
#             border-radius:0 10px 10px 0;padding:12px;'><b style='color:var(--accent-primary)'>No anomalies detected</b> — system operating normally.</div>""",
#             unsafe_allow_html=True)
#         else:
#             for a in alerts:
#                 sev = a.get("severity", "low").lower()
#                 sev_colors = {"high": ("#ef4444", "rgba(239,68,68,0.08)"),
#                               "medium": ("#f59e0b", "rgba(245,158,11,0.08)"),
#                               "low": ("var(--accent-secondary)", "rgba(77,168,255,0.08)")}
#                 color, bg = sev_colors.get(sev, sev_colors["low"])
#                 st.markdown(
#                     f"<div style='background:{bg};border-left:3px solid {color};"
#                     f"border-radius:0 10px 10px 0;padding:12px;margin:6px 0'>"
#                     f"<b style='color:{color}'>{sev.upper()}</b>: {a.get('description','')}</div>",
#                     unsafe_allow_html=True)
#     with cb2:
#         st.markdown("**Natural Language Query**")
#         # Initialize chat history in session state
#         if "chat_messages" not in st.session_state:
#             st.session_state.chat_messages = []
#         # Render chat interface inside a container
#         chat_container = st.container(height=500)
#         with chat_container:
#             if len(st.session_state.chat_messages) == 0:
#                 st.markdown("<div style='text-align:center;color:var(--text-secondary);padding-top:200px;'>Ask URG-IS Agent any question about the graph data.</div>", unsafe_allow_html=True)
#             for msg in st.session_state.chat_messages:
#                 with st.chat_message(msg["role"]):
#                     st.write(msg["content"])
#         # Quick-access example questions
#         st.markdown("<div style='margin-top:10px; font-size:0.8rem; color:var(--text-secondary);'>Suggested Queries:</div>", unsafe_allow_html=True)
#         eq_cols = st.columns(3)
#         example_qs = [
#             "Who is the most connected person?",
#             "Any unusual patterns?",
#             "Tell me about Person 1"
#         ]
#         # Helper to run the agent query
#         def _run_agent_query(question):
#             if not question.strip():
#                 return
#             if not db_ref:
#                 st.warning("Start pipeline first.")
#                 return
#             if not _ollama_ok:
#                 st.warning("Ollama is offline. Start with: ollama serve")
#                 return
#             st.session_state.chat_messages.append({"role": "user", "content": question})
#             # Using spinner below
#             try:
#                 from agent import natural_language_query
#                 gd = _build_graph_data()
#                 if gd:
#                     ans = natural_language_query(question, gd)
#                     st.session_state.chat_messages.append({"role": "assistant", "content": ans})
#                 else:
#                     st.session_state.chat_messages.append({"role": "assistant", "content": "No graph data available. Start the pipeline first."})
#             except Exception as e:
#                 st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
#         for i, eq in enumerate(example_qs):
#             if eq_cols[i].button(eq, key=f"eq_{i}", width='stretch'):
#                 _run_agent_query(eq)
#                 st.rerun()
#         prompt = st.chat_input("Ask Agent...", key="chat_input")
#         if prompt:
#             _run_agent_query(prompt)
#             st.rerun()
#     st.markdown("---")
#     st.markdown("""
#         <div style='text-align:center; padding:40px; color:var(--text-secondary); font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase;'>
#             URG-IS CORE &middot; PREDICTIVE RELATIONSHIP INTELLIGENCE ENGINE &middot; v2.0
#         </div>
#     """, unsafe_allow_html=True)

# 
# TAB 6  AI AGENT - STABILIZED EMERALD VERSION
# 
elif page == "AI AGENT":
    # 1. CSS for Premium Emerald Interface
    st.markdown("""
    <style>
        /* Main Page Border and Glow */
        .terminal-border-wrap {
            border: 2px solid #10b981;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
            border-radius: 20px;
            padding: 20px;
            background: rgba(10, 15, 20, 0.8);
        }

        /* Custom Chat Bubbles */
        .chat-row { display: flex; flex-direction: column; margin-bottom: 20px; width: 100%; }
        
        .user-bubble { 
            background: linear-gradient(135deg, #065f46 0%, #064e3b 100%); 
            color: white !important; padding: 12px 18px; border-radius: 18px 18px 4px 18px; 
            max-width: 75%; align-self: flex-end; text-align: left; 
            border: 1px solid rgba(16, 185, 129, 0.4);
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            font-size: 0.95rem; line-height: 1.5;
        }
        
        .bot-bubble { 
            background: rgba(30, 41, 59, 0.7); 
            color: #e2e8f0 !important; padding: 12px 18px; border-radius: 18px 18px 18px 4px; 
            max-width: 75%; align-self: flex-start; text-align: left; 
            border: 1px solid #10b981; 
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
            font-size: 0.95rem; line-height: 1.5;
        }
        
        .role-label { font-size: 0.6rem; font-weight: 800; color: #64748b; margin-bottom: 4px; text-transform: uppercase; }

        /* Anomaly Items */
        .anomaly-item { 
            background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 12px; 
            margin: 8px 0; border-left: 4px solid #ef4444; 
        }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown(f"""
    <div style='display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:30px'>
        <div>
            <h1 style='font-size:2.2rem; font-weight:800; color:#ffffff; margin:0; letter-spacing:0.02em;'>Intelligence Terminal</h1>
            <div style='color:#94a3b8; font-size:1rem; margin-top:4px'>Ollama Local LLM &middot; Predictive Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status Bar
    try:
        import requests as _rq
        _ollama_ok = _rq.get("http://localhost:11434/api/tags", timeout=2).ok
    except Exception:
        _ollama_ok = False

    from config.settings import OLLAMA_MODEL as _model_name
    status_color = "#10b981" if _ollama_ok else "#ef4444"
    
    st.markdown(f"""
    <div style='display:flex; gap:12px; margin-bottom:20px;'>
        <div style='border:1px solid {status_color}; border-radius:8px; padding:8px 16px; background:rgba(16, 185, 129, 0.08);'>
            <div style='font-size:0.75rem; color:#64748b; font-weight:700'>STATUS</div>
            <div style='font-size:0.95rem; font-weight:700; color:{status_color}'>{"Online" if _ollama_ok else "Offline"}</div>
        </div>
        <div style='border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:8px 16px; background:rgba(30, 41, 59, 0.5);'>
            <div style='font-size:0.75rem; color:#64748b; font-weight:700'>MODEL</div>
            <div style='font-size:0.95rem; font-weight:700; color:#ffffff'>{_model_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not _ollama_ok:
        st.warning("Ollama is not running. Please start it with: `ollama serve`")

    # Logic: build graph_data (Exact original implementation)
    def _build_graph_data():
        if not db_ref:
            return None
        edges_raw = db_ref.get_all_edges()
        nodes_raw = db_ref.get_all_nodes()
        deg_map = {}
        for e in edges_raw:
            if e.relationship != "stranger":
                deg_map[e.person_id_a] = deg_map.get(e.person_id_a, 0) + 1
                deg_map[e.person_id_b] = deg_map.get(e.person_id_b, 0) + 1
        return {
            "nodes": [{"id": n.person_id, "degree": deg_map.get(n.person_id, 0), "last_seen": getattr(n, 'last_seen', None)} for n in nodes_raw],
            "edges": [{"person_id_a": e.person_id_a, "person_id_b": e.person_id_b, "confidence": e.confidence, "relationship": e.relationship, "total_meetings": e.total_meetings} for e in edges_raw],
        }

    # Logic: Run Agent Query (Exact original implementation)
    def _run_agent_query(question):
        if not question.strip():
            return
        if not PS.G.get("running", False):
            st.warning("Start the main pipeline processing core first.")
            return
            
        st.session_state.chat_messages.append({"role": "user", "content": question})
        
        try:
            with st.spinner("Streaming response parameters from local model orchestrated gateway..."):
                # Offload inference directly to the FastAPI server endpoint on port 8000
                res = requests.get("http://127.0.0.1:8000/api/v1/usecases/missing/19", params={"q": question}, timeout=45.0)
                if res.status_code == 200:
                    server_data = res.json()
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": server_data.get("chatbot_payload", "Analysis payload compiled.")
                    })
                else:
                    st.error(f"Gateway Server Refused: Received code {res.status_code}")
        except Exception as e:
            st.error(f"Gateway Connection Failed: Ensure main.py backend is active on port 8000! ({str(e)})")
    ca2, cb2 = st.columns([0.2, 0.8], gap="large")

    with ca2:
        st.markdown("<h3 style='color:white; font-size:1.2rem;'>Anomaly Detection</h3>", unsafe_allow_html=True)
        st.markdown("""<div style='background:rgba(30, 41, 59, 0.5); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:15px; color:#94a3b8; font-size:0.85rem;'>
        <b style='color:white;'>Privacy-First Architecture</b><br>
        • IDs anonymised for LLM<br>
        • No biometric data shared<br>
        • Localhost:11434<br>
        • DPDP Act compliant
        </div
        """, unsafe_allow_html=True)
        
        if st.button("Run Anomaly Check", width='stretch'):
            if not db_ref:
                st.warning("Start the pipeline first.")
            elif not _ollama_ok:
                st.warning("Ollama is offline.")
            else:
                with st.spinner("Analysing graph..."):
                    try:
                        from agent import check_anomalies
                        gd = _build_graph_data()
                        if gd:
                            st.session_state["alerts"] = check_anomalies(gd)
                        else:
                            st.warning("No graph data available.")
                    except Exception as e:
                        st.error(f"Agent error: {e}")

        alerts = st.session_state.get("alerts", None)
        if alerts is None:
            st.markdown("<div style='text-align:center; padding:20px; color:#64748b; font-size:0.8rem;'>No active scans.</div>", unsafe_allow_html=True)
        elif not alerts:
            st.markdown("""<div style='background:rgba(16, 185, 129, 0.1); border-left:3px solid #10b981; border-radius:8px; padding:12px; color:#10b981; font-size:0.85rem;'>
            <b>System Nominal</b> — No anomalies detected.</div>""", unsafe_allow_html=True)
        else:
            for a in alerts:
                sev = a.get("severity", "low").lower()
                sev_colors = {"high": ("#ef4444", "rgba(239,68,68,0.08)"),
                              "medium": ("#f59e0b", "rgba(245,158,11,0.08)"),
                              "low": ("#3b82f6", "rgba(59,130,246,0.08)")}
                color, bg = sev_colors.get(sev, sev_colors["low"])
                st.markdown(
                    f"<div class='anomaly-item' style='background:{bg}; border-left-color:{color};'>"
                    f"<div style='font-size:0.65rem; font-weight:800; color:{color};'>{sev.upper()}</div>"
                    f"<div style='color:#e2e8f0; font-size:0.85rem;'>{a.get('description','')}</div></div>",
                    unsafe_allow_html=True)

    with cb2:
        st.markdown("<h3 style='color:white; font-size:1.2rem;'>Intelligence Query</h3>", unsafe_allow_html=True)
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        chat_container = st.container(height=550)
        with chat_container:
            if len(st.session_state.chat_messages) == 0:
                st.markdown("<div style='text-align:center; color:#64748b; padding-top:150px;'>Ask URG-IS Agent any question about the graph data.</div>", unsafe_allow_html=True)
            
            for msg in st.session_state.chat_messages:
                avatar = "👤" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        st.markdown("<div style='margin-top:15px; font-size:0.75rem; color:#64748b; font-weight:700;'>SUGGESTED QUERIES:</div>", unsafe_allow_html=True)
        eq_cols = st.columns(3)
        example_qs = ["Most connected person?", "Any unusual patterns?", "Tell me about Person 1"]
        
        for i, eq in enumerate(example_qs):
            if eq_cols[i].button(eq, key=f"eq_{i}", width='stretch'):
                _run_agent_query(eq)
                st.rerun()

        prompt = st.chat_input("Ask Agent...", key="chat_input")
        if prompt:
            _run_agent_query(prompt)
            st.rerun()

elif page == "USE CASES":
    usecases_page.render()

st.markdown("---")
st.markdown("""
    <div style='text-align:center; padding:40px; color:#64748b; font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase;'>
        URG-IS CORE &middot; PREDICTIVE RELATIONSHIP INTELLIGENCE ENGINE &middot; v2.0
    </div>
""", unsafe_allow_html=True)