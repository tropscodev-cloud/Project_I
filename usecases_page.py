import os
import sys
import time
import requests
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pipeline_state as PS
import re

# Maintain clean project scope lookup tree variables
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ─── TELEMETRY TEXT SANITIZATION UTILITIES ───
def clean_response_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("*", "")
    emoji_pattern = re.compile(
        u"[\U0001F600-\U0001F64F|\U0001F300-\U0001F5FF|\U0001F680-\U0001F6FF|\U0001F1E0-\U0001F1FF|\u200d|\ufe0f]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r"", text).strip()

@st.cache_resource
def load_predictive_agents():
    from Agents.movement import ZeroShotMovementAgent
    from Agents.anomaly import BehavioralAnomalyAgent
    return ZeroShotMovementAgent(prediction_seconds=5), BehavioralAnomalyAgent(anomaly_sensitivity_threshold=3.0)

@st.cache_resource
def load_graph_database():
    from core.graph.graph_db import GraphDB
    return GraphDB(snapshot_path="data/snapshots/prod_graph.json")

# ─── MASTER LAYOUT PANEL ───
def render():
    st.markdown("""
        <style>
        .reportview-container { background: #071018; color: #F8FAFC; }
        [data-testid="stAppViewContainer"] { background-color: #071018 !important; }
        .home-style-card { background: rgba(15, 23, 35, 0.8) !important; backdrop-filter: blur(25px) !important; border: 1px solid rgba(255, 255, 255, 0.05) !important; border-radius: 24px !important; padding: 40px !important; display: flex !important; gap: 40px !important; margin-bottom: 30px !important; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5) !important; }
        .card-left-col { flex: 0 0 160px !important; display: flex !important; flex-direction: column !important; align-items: center !important; border-right: 1px solid rgba(255, 255, 255, 0.05) !important; padding-right: 40px !important; }
        .badge-circle { width: 100px !important; height: 100px !important; border-radius: 50% !important; display: flex !important; align-items: center !important; justify-content: center !important; font-size: 1.3rem !important; font-weight: 800 !important; margin-bottom: 20px !important; background: rgba(15, 23, 35, 0.8) !important; position: relative !important; }
        .badge-circle-sec { border: 2px solid #10B981 !important; box-shadow: 0 0 25px rgba(16, 185, 129, 0.25); }
        .badge-circle-wel { border: 2px solid #06B6D4 !important; box-shadow: 0 0 25px rgba(6, 182, 212, 0.25); }
        .chat-container-console { background: rgba(8, 14, 22, 0.6) !important; border: 1px solid rgba(255, 255, 255, 0.04) !important; border-radius: 14px !important; padding: 16px !important; margin-bottom: 20px !important; }
        .chat-bubble-pill-bot { background: rgba(15, 23, 35, 0.7) !important; border: 1px solid rgba(255, 255, 255, 0.05) !important; color: #E2E8F0 !important; padding: 12px 18px !important; border-radius: 20px !important; font-size: 0.88rem !important; }
        .chat-bubble-pill-user { background: linear-gradient(135deg, rgba(6, 78, 59, 0.4) 0%, rgba(2, 44, 34, 0.6) 100%) !important; border: 1px solid rgba(16, 185, 129, 0.25) !important; color: #F8FAFC !important; padding: 12px 18px !important; border-radius: 20px !important; font-size: 0.88rem !important; max-width: 85% !important; margin-left: auto; margin-bottom: 12px !important; }
        .chat-visual-card { background: rgba(7, 16, 24, 0.4) !important; border: 1px solid rgba(255, 255, 255, 0.05) !important; border-radius: 16px !important; padding: 16px !important; min-height: 320px !important; display: flex !important; flex-direction: column !important; justify-content: center !important; }
        </style>
    """, unsafe_allow_html=True)

    db = PS.G.get("graph_db")
    if db is None:
        db = load_graph_database()

    if "active_subcase" not in st.session_state:
        st.session_state.active_subcase = "None"
        
    if "usecase_chats" not in st.session_state:
        st.session_state.usecase_chats = {
            "sec_sleeper": [{"role": "assistant", "content": "Sleeper Cell Patterns Agent online over port 8000."}],
            "sec_associates": [{"role": "assistant", "content": "Hidden Associates Agent active over port 8000."}],
            "wel_missing": [{"role": "assistant", "content": "Missing Child Search Desk synchronized. Select parameters to track."}],
            "wel_density": [{"role": "assistant", "content": "Volumetric Crowd Density Monitor armed."}]
        }
    if "welfare_payload" not in st.session_state: st.session_state.welfare_payload = {}
    if "security_payload" not in st.session_state: st.session_state.security_payload = {}

    # MAIN MASTER COMMAND GRID
    if st.session_state.active_subcase == "None":
        st.markdown("<h2 style='text-align:center; margin-bottom: 40px;'>URG-IS Prototype Use Case Selection Grid</h2>", unsafe_allow_html=True)
        
        # HUB PANEL 1: HIGH STAKES SECURITY
        st.markdown('<div class="home-style-card">', unsafe_allow_html=True)
        col_s1_left, col_s1_right = st.columns([1, 4])
        with col_s1_left:
            st.markdown('<div class="card-left-col"><div class="badge-circle badge-circle-sec">SEC</div></div>', unsafe_allow_html=True)
        with col_s1_right:
            st.markdown('<h4>Universal High-Stakes Threat Intelligence</h4>', unsafe_allow_html=True)
            cols_grid = st.columns(2)
            with cols_grid[0]:
                if st.button("Sleeper Cell Patterns", key="btn_sec_sleeper", use_container_width=True):
                    st.session_state.active_subcase = "sec_sleeper"; st.rerun()
            with cols_grid[1]:
                if st.button("Hidden Associates Mapping", key="btn_sec_associates", use_container_width=True):
                    st.session_state.active_subcase = "sec_associates"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # HUB PANEL 2: PUBLIC WELFARE
        st.markdown('<div class="home-style-card">', unsafe_allow_html=True)
        col_s2_left, col_s2_right = st.columns([1, 4])
        with col_s2_left:
            st.markdown('<div class="card-left-col"><div class="badge-circle badge-circle-wel">WEL</div></div>', unsafe_allow_html=True)
        with col_s2_right:
            st.markdown('<h4>Distributed Public Welfare & Crowd Safety Command</h4>', unsafe_allow_html=True)
            cols_grid2 = st.columns(2)
            with cols_grid2[0]:
                if st.button("Missing Child Tracing Desk", key="btn_wel_missing", use_container_width=True):
                    st.session_state.active_subcase = "wel_missing"; st.rerun()
            with cols_grid2[1]:
                if st.button("Choke Point Volume Trends", key="btn_wel_density", use_container_width=True):
                    st.session_state.active_subcase = "wel_density"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    sub = st.session_state.active_subcase
    is_sec = sub.startswith("sec")
    is_wel = sub.startswith("wel")
    
    if st.button("← Return to Master Context Hub", key="back_to_main_panel"):
        st.session_state.active_subcase = "None"; st.rerun()

    st.markdown(f"<h3 style='text-align:center;'>WORKSPACE DESK: {sub.upper().replace('_', ' // ')}</h3>", unsafe_allow_html=True)

    st.markdown('<div class="home-style-card">', unsafe_allow_html=True)
    col_console_left, col_console_right = st.columns([1.35, 1.0], gap="large")
    
    with col_console_left:
        chat_container = st.container(height=380)
        with chat_container:
            for msg in st.session_state.usecase_chats[sub]:
                avatar = "👤" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        # SELECTION MANAGEMENT BLOCK
        st.markdown('<div style="margin-top: 15px;">', unsafe_allow_html=True)
        if sub == "sec_sleeper":
            all_nodes = db.get_all_nodes()
            pids = sorted([n.person_id for n in all_nodes]) if all_nodes else ["42"]
            selected_pid = st.selectbox("Anchor Target Profile ID", pids, key="sel_node_sec_s")
            if st.button("Analyze Sleeper Modularity Group", use_container_width=True):
                submit_security_query(f"http://127.0.0.1:8000/api/v1/usecases/security/sleeper/{selected_pid}", "security_payload", sub, f"Audit sleeper partitions for P{selected_pid}")
                st.rerun()

        elif sub == "sec_associates":
            all_nodes = db.get_all_nodes()
            pids = sorted([n.person_id for n in all_nodes]) if all_nodes else ["42"]
            selected_pid = st.selectbox("Anchor Target Profile ID", pids, key="sel_node_sec_a")
            if st.button("Extract Co-Presence Proximity Clustered Rings", use_container_width=True):
                submit_security_query(f"http://127.0.0.1:8000/api/v1/usecases/security/associates/{selected_pid}", "security_payload", sub, f"Trace hidden accomplices of P{selected_pid}")
                st.rerun()

        elif sub == "wel_missing":
            all_nodes = db.get_all_nodes()
            pids = sorted([n.person_id for n in all_nodes]) if all_nodes else ["19"]
            selected_pid = st.selectbox("Missing Subject Target ID", pids, key="sel_node_wel_m")
            if st.button("Generate Kinematic Trajectory Matrix Path", use_container_width=True):
                submit_welfare_query(f"http://127.0.0.1:8000/api/v1/usecases/missing/{selected_pid}", "welfare_payload", sub, f"Forecasting microtrace metrics for Child P{selected_pid}")
                st.rerun()

        elif sub == "wel_density":
            selected_cam = st.selectbox("Target Camera Sensor Portal", ["cam1", "cam2", "cam3"], key="sel_node_wel_d")
            if st.button("Analyze Volumetric Flow Trends", use_container_width=True):
                submit_welfare_query(f"http://127.0.0.1:8000/api/v1/usecases/crowd/{selected_cam}", "welfare_payload", sub, f"Evaluate crowd bottlenecks at sector {selected_cam}")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # USER NARRATIVE CHAT ENTRY BLOCK
        prompt = st.chat_input("Interact with active use case model arrays...")
        if prompt:
            if sub == "sec_sleeper":
                pid = st.session_state.get("sel_node_sec_s", "42")
                submit_security_query(f"http://127.0.0.1:8000/api/v1/usecases/security/sleeper/{pid}", "security_payload", sub, prompt)
            elif sub == "sec_associates":
                pid = st.session_state.get("sel_node_sec_a", "42")
                submit_security_query(f"http://127.0.0.1:8000/api/v1/usecases/security/associates/{pid}", "security_payload", sub, prompt)
            elif sub == "wel_missing":
                pid = st.session_state.get("sel_node_wel_m", "19")
                submit_welfare_query(f"http://127.0.0.1:8000/api/v1/usecases/missing/{pid}", "welfare_payload", sub, prompt)
            elif sub == "wel_density":
                cam = st.session_state.get("sel_node_wel_d", "cam1")
                submit_welfare_query(f"http://127.0.0.1:8000/api/v1/usecases/crowd/{cam}", "welfare_payload", sub, prompt)
            st.rerun()

    with col_console_right:
        st.markdown('<div class="chat-visual-card">', unsafe_allow_html=True)
        has_payload = False
        
        if is_sec and st.session_state.security_payload.get("status") == "SUCCESS":
            has_payload = True
            payload = st.session_state.security_payload
            st.markdown(f"<div style='padding:12px; background:rgba(239,68,68,0.08); border:1px solid #EF4444; border-radius:10px; margin-bottom:12px;'><b style='color:#EF4444;'>THREAT STATUS VERDICT: {payload.get('threat_classification', 'GREEN_MONITORING')}</b></div>", unsafe_allow_html=True)
            relations = []
            graph_data = db.get_person_graph(payload.get('target_id', '42'))
            if graph_data:
                for conn in graph_data.get("connections", []):
                    relations.append({"person_id": conn["person_id"], "confidence": conn["confidence"], "relationship": conn["relationship"], "color": conn["color"]})
            fig_sec = plot_security_network(payload.get('target_id', '42'), relations[:6])
            if fig_sec: st.plotly_chart(fig_sec, use_container_width=True)
                
        elif is_wel and st.session_state.welfare_payload.get("status") == "SUCCESS":
            has_payload = True
            payload = st.session_state.welfare_payload
            if "predictions" in payload:
                fig_wel = plot_welfare_path(payload, payload.get("target_id", "19"))
                st.plotly_chart(fig_wel, use_container_width=True)
            elif "metrics" in payload:
                fig_crowd = plot_crowd_density(payload)
                st.plotly_chart(fig_crowd, use_container_width=True)

        if not has_payload:
            st.markdown('<div style="text-align:center; color:#64748b; font-family:monospace; padding-top:120px;">TELEMETRY STREAM INACTIVE<br>[Fire API execution command packet]</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── UNBLOCKED REST DISPATCH CALLERS ───
def submit_security_query(url: str, state_key: str, subcase: str, query: str):
    if not any(msg["content"] == query for msg in st.session_state.usecase_chats[subcase]):
        st.session_state.usecase_chats[subcase].append({"role": "user", "content": query})
    try:
        with st.spinner("Streaming response payload packet array from local model orchestrator..."):
            res = requests.get(url, params={"q": query}, timeout=45.0)
            if res.status_code == 200:
                data = res.json()
                st.session_state[state_key] = data
                st.session_state.usecase_chats[subcase].append({"role": "assistant", "content": clean_response_text(data.get("chatbot_payload", "Trace finalized."))})
            else:
                st.error(f"Gateway Error Boundary Broken: Recieved status {res.status_code}")
    except Exception as e:
        st.error(f"Microservice Connection Failed: Ensure main.py backend is active on port 8000! ({str(e)})")

def submit_welfare_query(url: str, state_key: str, subcase: str, query: str):
    if not any(msg["content"] == query for msg in st.session_state.usecase_chats[subcase]):
        st.session_state.usecase_chats[subcase].append({"role": "user", "content": query})
    try:
        with st.spinner("Streaming response payload packet array from local model orchestrator..."):
            res = requests.get(url, params={"q": query}, timeout=45.0)
            if res.status_code == 200:
                data = res.json()
                st.session_state[state_key] = data
                st.session_state.usecase_chats[subcase].append({"role": "assistant", "content": clean_response_text(data.get("chatbot_payload", "Metrics compiled."))})
            else:
                st.error(f"Gateway Error Boundary Broken: Recieved status {res.status_code}")
    except Exception as e:
        st.error(f"Microservice Connection Failed: Ensure main.py backend is active on port 8000! ({str(e)})")

# ─── COMPONENT CHART RENDER PLOTS ───
def plot_security_network(target_id, connections):
    fig = go.Figure()
    N = len(connections)
    if N == 0: return None
    for i, conn in enumerate(connections):
        angle = 2 * np.pi * i / N
        x_pos, y_pos = 4 * np.cos(angle), 4 * np.sin(angle)
        fig.add_trace(go.Scatter(x=[0, x_pos], y=[0, y_pos], mode="lines", line=dict(color=conn.get("color", "#4DA8FF"), width=3), showlegend=False, hoverinfo="none"))
    node_x, node_y, node_text, node_colors = [], [], [], []
    for i, conn in enumerate(connections):
        angle = 2 * np.pi * i / N
        node_x.append(4 * np.cos(angle)); node_y.append(4 * np.sin(angle))
        node_text.append(f"P{conn['person_id']}"); node_colors.append(conn.get("color", "#4DA8FF"))
    node_x.append(0); node_y.append(0); node_text.append(f"TARGET: P{target_id}"); node_colors.append("#EF4444")
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", marker=dict(size=20, color=node_colors), text=node_text, textposition="top center", showlegend=False))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=False, showticklabels=False), margin=dict(l=10,r=10,t=10,b=10), height=280)
    return fig

def plot_welfare_path(payload, target_id):
    fig_track = go.Figure()
    preds = payload.get("predictions", {})
    cx, cy = preds.get("last_seen", (5.0, 5.0))
    fig_track.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers+text", marker=dict(size=14, color="#06B6D4"), text=[f"LAST SEEN: P{target_id}"], textposition="top right"))
    future_path = preds.get("projected_path_trail", [])
    if future_path:
        fx, fy = zip(*future_path)
        fig_track.add_trace(go.Scatter(x=fx, y=fy, mode="lines+markers", line=dict(dash="dot", color="#F59E0B", width=2.5)))
    fig_track.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=10,r=10,t=10,b=10))
    return fig_track

def plot_crowd_density(payload):
    fig = go.Figure()
    cam_id = payload.get("camera_id", "cam1").upper()
    curr_count = payload.get("metrics", {}).get("occupancy_density_count", 0)
    fig.add_annotation(x=5, y=5.5, text=f"<b>Gateway Telemetry Stream: {cam_id} Active</b><br>Tracked Volumetric Headcount: {int(curr_count)} targets", showarrow=False, font=dict(color="#E2E8F0", size=14))
    fig.update_layout(plot_bgcolor="rgba(5, 12, 26, 1)", paper_bgcolor="rgba(0,0,0,0)", height=340, xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=False, showticklabels=False))
    return fig