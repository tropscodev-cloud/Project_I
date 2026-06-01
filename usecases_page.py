import os
import time
import requests
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pipeline_state as PS

@st.cache_resource
def load_predictive_agents():
    from Agents.movement import ZeroShotMovementAgent
    from Agents.anomaly import BehavioralAnomalyAgent
    return ZeroShotMovementAgent(prediction_seconds=5), BehavioralAnomalyAgent(anomaly_sensitivity_threshold=3.0)

@st.cache_resource
def load_graph_database():
    from core.graph.graph_db import GraphDB
    return GraphDB(snapshot_path="data/snapshots/prod_graph.json")

def render():
    st.markdown("""
        <style>
        .reportview-container { background: #071018; color: #F8FAFC; }
        [data-testid="stAppViewContainer"] { background-color: #071018 !important; }
        
        .home-style-card {
            background: rgba(15, 23, 35, 0.8) !important;
            backdrop-filter: blur(25px) !important;
            -webkit-backdrop-filter: blur(25px) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 24px !important;
            padding: 40px !important;
            display: flex !important;
            gap: 40px !important;
            margin-bottom: 30px !important;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5) !important;
        }
        
        .card-left-col {
            flex: 0 0 160px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
            padding-right: 40px !important;
        }
        
        .card-right-col {
            flex: 1 !important;
        }
        
        .badge-circle {
            width: 100px !important;
            height: 100px !important;
            border-radius: 50% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 2.2rem !important;
            margin-bottom: 20px !important;
            background: rgba(15, 23, 35, 0.8) !important;
            position: relative !important;
        }
        
        .badge-circle-sec {
            border: 2px solid #10B981 !important;
            box-shadow: 0 0 25px rgba(16, 185, 129, 0.25), inset 0 0 15px rgba(16, 185, 129, 0.15) !important;
        }
        .badge-circle-wel {
            border: 2px solid #06B6D4 !important;
            box-shadow: 0 0 25px rgba(6, 182, 212, 0.25), inset 0 0 15px rgba(6, 182, 212, 0.15) !important;
        }
        .badge-circle-civ {
            border: 2px solid #F59E0B !important;
            box-shadow: 0 0 25px rgba(245, 158, 11, 0.25), inset 0 0 15px rgba(245, 158, 11, 0.15) !important;
        }

        .mini-stats-grid {
            display: flex !important;
            flex-direction: column !important;
            gap: 10px !important;
            width: 100% !important;
        }
        .mini-stat-item {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.04) !important;
            border-radius: 8px !important;
            padding: 6px 10px !important;
            text-align: center !important;
        }
        .mini-stat-val {
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 800 !important;
            font-size: 1.1rem !important;
            color: #F8FAFC !important;
        }
        .mini-stat-lbl {
            font-size: 0.65rem !important;
            color: #8FA1B7 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin-top: 2px !important;
        }
        
        .usecase-subheading {
            font-size: 0.78rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.35em !important;
            text-transform: uppercase !important;
            color: #4DA8FF !important;
            margin-bottom: 12px !important;
        }
        .usecase-title-text {
            font-size: 2.2rem !important;
            font-weight: 900 !important;
            margin: 0 0 20px 0 !important;
            color: #F8FAFC !important;
        }
        .usecase-desc-text {
            font-size: 1.1rem !important;
            line-height: 1.8 !important;
            color: #A8B6C7 !important;
            margin-bottom: 25px !important;
        }
        
        .active-agent-btn div[data-testid="stButton"] button {
            background: rgba(16, 185, 129, 0.08) !important;
            border: 1px solid rgba(16, 185, 129, 0.3) !important;
            color: #7CFFB2 !important;
            border-radius: 8px !important;
            font-size: 0.78rem !important;
            padding: 10px 14px !important;
            font-weight: 600 !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.05) !important;
            white-space: pre-wrap !important;
            text-align: center !important;
        }
        .active-agent-btn div[data-testid="stButton"] button:hover {
            background: rgba(16, 185, 129, 0.18) !important;
            border-color: #10B981 !important;
            color: #FFFFFF !important;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.2) !important;
            transform: translateY(-2px) !important;
        }
        
        .locked-chip {
            background: rgba(255, 255, 255, 0.015) !important;
            border: 1px solid rgba(255, 255, 255, 0.03) !important;
            color: #64748b !important;
            border-radius: 8px !important;
            padding: 10px 14px !important;
            font-size: 0.78rem !important;
            text-align: center !important;
            font-weight: 600 !important;
            cursor: not-allowed !important;
            opacity: 0.55 !important;
            height: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .agent-title-block {
            text-align: center;
            margin-bottom: 25px;
        }
        .agent-title {
            font-size: 2.1rem !important;
            font-weight: 900 !important;
            color: #F8FAFC !important;
        }
        .agent-subtitle {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.8rem !important;
            color: #7CFFB2 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.15em !important;
            margin-top: 4px !important;
        }

        .agent-box {
            background: rgba(15, 23, 35, 0.8) !important;
            backdrop-filter: blur(25px) !important;
            -webkit-backdrop-filter: blur(25px) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5) !important;
            border-radius: 20px !important;
            padding: 24px !important;
            margin-bottom: 25px !important;
        }

        .chat-container-console {
            background: rgba(8, 14, 22, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.04) !important;
            border-radius: 14px !important;
            padding: 16px !important;
            margin-bottom: 20px !important;
        }
        
        .chat-row-horizontal {
            display: flex !important;
            align-items: flex-start !important;
            gap: 14px !important;
            width: 100% !important;
            margin-bottom: 16px !important;
        }
        .chat-avatar-col {
            flex: 0 0 45px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }
        .chat-avatar-icon {
            width: 36px !important;
            height: 36px !important;
            border-radius: 50% !important;
            background: rgba(124, 255, 178, 0.08) !important;
            border: 1px solid rgba(124, 255, 178, 0.25) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 1.05rem !important;
            box-shadow: 0 0 8px rgba(124, 255, 178, 0.08) !important;
        }
        .chat-avatar-label {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.55rem !important;
            font-weight: 800 !important;
            color: #64748b !important;
            margin-top: 3px !important;
            text-align: center !important;
        }
        
        .chat-bubble-col {
            flex: 1 !important;
        }
        .chat-bubble-pill-bot {
            background: rgba(15, 23, 35, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            color: #E2E8F0 !important;
            padding: 12px 18px !important;
            border-radius: 20px !important;
            font-size: 0.88rem !important;
            line-height: 1.5 !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2) !important;
        }
        .chat-bubble-pill-user {
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.4) 0%, rgba(2, 44, 34, 0.6) 100%) !important;
            border: 1px solid rgba(16, 185, 129, 0.25) !important;
            color: #F8FAFC !important;
            padding: 12px 18px !important;
            border-radius: 20px !important;
            font-size: 0.88rem !important;
            line-height: 1.5 !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.15) !important;
            max-width: 85% !important;
            margin-left: auto !important;
            margin-bottom: 12px !important;
        }

        .chat-visual-card {
            background: rgba(7, 16, 24, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 16px !important;
            padding: 16px !important;
            box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.3) !important;
            height: 100% !important;
            min-height: 320px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
        }

        .back-btn button {
            background: rgba(15, 23, 35, 0.6) !important;
            color: #AAB6C5 !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 8px !important;
            padding: 6px 14px !important;
            font-size: 0.78rem !important;
            transition: all 0.2s ease !important;
        }
        .back-btn button:hover {
            color: #F8FAFC !important;
            border-color: rgba(124, 255, 178, 0.3) !important;
            background: rgba(124, 255, 178, 0.03) !important;
        }
        
        .sugg-btn button {
            background: rgba(16, 185, 129, 0.04) !important;
            border: 1px solid rgba(16, 185, 129, 0.2) !important;
            color: #7CFFB2 !important;
            border-radius: 18px !important;
            font-size: 0.78rem !important;
            padding: 6px 16px !important;
            transition: all 0.2s ease !important;
        }
        .sugg-btn button:hover {
            background: rgba(16, 185, 129, 0.12) !important;
            border-color: #10B981 !important;
            color: #FFFFFF !important;
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.15) !important;
            transform: translateY(-2px) !important;
        }

        .civic-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }
        .civic-table th {
            text-align: left;
            padding: 8px 10px;
            color: #64748b;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            letter-spacing: 0.05em;
        }
        .civic-table td {
            padding: 8px 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }
        .status-pill {
            font-size: 0.68rem;
            font-weight: 800;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
        }
        .status-fine { background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.25); }
        .status-flagged { background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.25); }
        .status-dispatched { background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.25); }
        
        [data-testid="stTabs"] { display: flex !important; }
        hr { border-color: rgba(255, 255, 255, 0.06) !important; margin: 15px 0 !important; }
        </style>
    """, unsafe_allow_html=True)

    db = PS.G.get("graph_db")
    if db is None:
        db = load_graph_database()

    if "active_subcase" not in st.session_state:
        st.session_state.active_subcase = "None"
        
    if "usecase_chats" not in st.session_state:
        st.session_state.usecase_chats = {
            "sec_sleeper": [{"role": "assistant", "content": "Sleeper Cell Patterns Agent online. Ready to evaluate loitering records and network graph structures for coordinated activity."}],
            "sec_terror": [{"role": "assistant", "content": "Anti-Terror Coordination Agent online. Analyzing potential tactical group actions and spatial proximity anomalies."}],
            "sec_network": [{"role": "assistant", "content": "Criminal Networks Agent active. Input target suspect ID to trace relations and partition clusters."}],
            "sec_associates": [{"role": "assistant", "content": "Hidden Associates Agent online. Ready to discover co-presence suspect clusters."}],
            
            "wel_missing": [{"role": "assistant", "content": "Missing Child Tracing Agent online. Select a Person ID below and click solve case to calculate trajectory forecasts."}],
            "wel_density": [{"role": "assistant", "content": "Crowd Density Monitoring Agent online. Scanning visual choke points for bottleneck hazards."}],
            "wel_stampede": [{"role": "assistant", "content": "Stampede Prevention Agent online. Monitoring crowd flow speeds and spatial compression risk levels."}],
            "wel_hotspots": [{"role": "assistant", "content": "Interaction Hotspots Agent active. Spotting density clusters and co-presence points in target sectors."}],
            
            "civ_traffic": [{"role": "assistant", "content": "Traffic Violations Agent online. Camera feed analysis armed. Ask me to scan for triple riding or helmet gaps."}],
            "civ_littering": [{"role": "assistant", "content": "Civic Violations compliance scanning active. Query camera locations for public littering alerts."}],
            "civ_behavior": [{"role": "assistant", "content": "Behavior Patterns Compliance Link active. Bridging municipal infractions to core warning logs."}],
            "civ_dispatch": [{"role": "assistant", "content": "Smart City Dispatch & Warnings Center online. Patrol officer tracking synced."}]
        }
    if "welfare_payload" not in st.session_state:
        st.session_state.welfare_payload = {}
    if "security_payload" not in st.session_state:
        st.session_state.security_payload = {}

    import glob
    img_files = glob.glob("/Users/bhargavi/.gemini/antigravity-ide/brain/*/security_center_*.png")
    image_path = img_files[0] if img_files else None

    if st.session_state.active_subcase == "None":
        st.markdown("<h2 style='text-align:center; margin-bottom: 40px;'>UseCases Tab</h2>", unsafe_allow_html=True)
        
        st.markdown('<div class="home-style-card">', unsafe_allow_html=True)
        col_s1_left, col_s1_right = st.columns([1, 4])
        with col_s1_left:
            st.markdown("""
                <div class="card-left-col">
                    <div class="badge-circle badge-circle-sec">🛡️</div>
                    <div class="mini-stats-grid">
                        <div class="mini-stat-item">
                            <div class="mini-stat-val">57</div>
                            <div class="mini-stat-lbl">Nodes</div>
                        </div>
                        <div class="mini-stat-item">
                            <div class="mini-stat-val">414</div>
                            <div class="mini-stat-lbl">Edges</div>
                        </div>
                        <div class="mini-stat-item">
                            <div class="mini-stat-val">11</div>
                            <div class="mini-stat-lbl">Cells</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_s1_right:
            st.markdown('<div class="card-right-col">', unsafe_allow_html=True)
            st.markdown('<div class="usecase-subheading">UNIVERSAL THREAT INTELLIGENCE</div>', unsafe_allow_html=True)
            st.markdown('<h2 class="usecase-title-text">High-Stakes Security</h2>', unsafe_allow_html=True)
            st.markdown('<p class="usecase-desc-text">From detecting Sleeper Cell patterns and Anti-Terror coordination to tracking Criminal Networks and identifying suspects\' hidden associates.</p>', unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.75rem; font-weight:800; color:#64748b; margin-top:20px; margin-bottom:12px; letter-spacing:0.1em;'>AVAILABLE INTELLIGENCE AGENTS:</div>", unsafe_allow_html=True)
            cols_grid = st.columns(4)
            with cols_grid[0]:
                st.markdown('<div class="active-agent-btn">', unsafe_allow_html=True)
                if st.button("🛡️ Sleeper Cell Patterns", key="btn_sec_sleeper"):
                    st.session_state.active_subcase = "sec_sleeper"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with cols_grid[1]:
                st.markdown('<div class="active-agent-btn">', unsafe_allow_html=True)
                if st.button("🛡️ Hidden Associates", key="btn_sec_associates"):
                    st.session_state.active_subcase = "sec_associates"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with cols_grid[2]:
                st.markdown('<div class="locked-chip">🔒 Anti-Terror Coord.</div>', unsafe_allow_html=True)
            with cols_grid[3]:
                st.markdown('<div class="locked-chip">🔒 Criminal Networks</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="home-style-card">', unsafe_allow_html=True)
        col_s2_left, col_s2_right = st.columns([1, 4])
        with col_s2_left:
            st.markdown("""
                <div class="card-left-col">
                    <div class="badge-circle badge-circle-wel">❤️</div>
                    <div class="mini-stats-grid">
                        <div class="mini-stat-item">
                            <div class="mini-stat-val">&lt; 1s</div>
                            <div class="mini-stat-lbl">Latency</div>
                        </div>
                        <div class="mini-stat-item">
                            <div class="mini-stat-val">98.4%</div>
                            <div class="mini-stat-lbl">Re-ID</div>
                        </div>
                        <div class="mini-stat-item">
                            <div class="mini-stat-val">07</div>
                            <div class="mini-stat-lbl">Feeds</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_s2_right:
            st.markdown('<div class="card-right-col">', unsafe_allow_html=True)
            st.markdown('<div class="usecase-subheading">NEURAL WELFARE LAYER</div>', unsafe_allow_html=True)
            st.markdown('<h2 class="usecase-title-text">Public Welfare</h2>', unsafe_allow_html=True)
            st.markdown('<p class="usecase-desc-text">We can trace a missing child across an entire city in seconds, pinpointing exactly who they were last seen with. We can monitor Crowd Density to prevent stampedes and identify interaction hotspots in real-time.</p>', unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.75rem; font-weight:800; color:#64748b; margin-top:20px; margin-bottom:12px; letter-spacing:0.1em;'>AVAILABLE INTELLIGENCE AGENTS:</div>", unsafe_allow_html=True)
            cols_grid2 = st.columns(4)
            with cols_grid2[0]:
                st.markdown('<div class="active-agent-btn">', unsafe_allow_html=True)
                if st.button("❤️ Missing Child Tracing", key="btn_wel_missing"):
                    st.session_state.active_subcase = "wel_missing"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with cols_grid2[1]:
                st.markdown('<div class="active-agent-btn">', unsafe_allow_html=True)
                if st.button("❤️ Crowd Density", key="btn_wel_density"):
                    st.session_state.active_subcase = "wel_density"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with cols_grid2[2]:
                st.markdown('<div class="locked-chip">🔒 Stampede Prev.</div>', unsafe_allow_html=True)
            with cols_grid2[3]:
                st.markdown('<div class="locked-chip">🔒 Interaction Hotspots</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="home-style-card">', unsafe_allow_html=True)
        col_s3_left, col_s3_right = st.columns([1, 4])
        with col_s3_left:
            st.markdown("""
                <div class="card-left-col">
                    <div class="badge-circle badge-circle-civ">🏙️</div>
                    <div class="mini-stats-grid">
                        <div class="mini-stat-item">
                            <div class="mini-stat-val">124</div>
                            <div class="mini-stat-lbl">Violations</div>
                        </div>
                        <div class="mini-stat-item">
                            <div class="mini-stat-val">94.2%</div>
                            <div class="mini-stat-lbl">Compliance</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_s3_right:
            st.markdown('<div class="card-right-col">', unsafe_allow_html=True)
            st.markdown('<div class="usecase-subheading">SMART CITY AUTOMATION</div>', unsafe_allow_html=True)
            st.markdown('<h2 class="usecase-title-text">Civic Order</h2>', unsafe_allow_html=True)
            st.markdown('<p class="usecase-desc-text">We automate the \'Smart City.\' From Traffic Violations like triple-riding and helmet gaps to Civic Violations like littering and public nuisance. We link these violations to behavior patterns, turning a simple fine into a tool for behavioral change.</p>', unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.75rem; font-weight:800; color:#64748b; margin-top:20px; margin-bottom:12px; letter-spacing:0.1em;'>AVAILABLE INTELLIGENCE AGENTS:</div>", unsafe_allow_html=True)
            cols_grid3 = st.columns(4)
            with cols_grid3[0]:
                st.markdown('<div class="active-agent-btn">', unsafe_allow_html=True)
                if st.button("🏙️ Traffic Violations", key="btn_civ_traffic"):
                    st.session_state.active_subcase = "civ_traffic"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with cols_grid3[1]:
                st.markdown('<div class="locked-chip">🔒 Civic Violations</div>', unsafe_allow_html=True)
            with cols_grid3[2]:
                st.markdown('<div class="locked-chip">🔒 Behavior Link</div>', unsafe_allow_html=True)
            with cols_grid3[3]:
                st.markdown('<div class="locked-chip">🔒 Dispatch Center</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    sub = st.session_state.active_subcase
    is_sec = sub.startswith("sec")
    is_wel = sub.startswith("wel")
    is_civ = sub.startswith("civ")
    
    agent_title = "Security Strategic Agent" if is_sec else "Welfare Strategic Agent" if is_wel else "Civic Strategic Agent"
    agent_subtitle = sub.replace("_", " // ").upper()

    col_back1, col_back2 = st.columns([1, 10])
    with col_back1:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back", key="back_to_main"):
            st.session_state.active_subcase = "None"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
        <div class="agent-title-block">
            <div class="agent-title">{agent_title}</div>
            <div class="agent-subtitle">{agent_subtitle}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="agent-box">', unsafe_allow_html=True)
    
    col_console_left, col_console_right = st.columns([1.35, 1.0], gap="large")
    
    with col_console_left:
        st.markdown('<div class="chat-container-console">', unsafe_allow_html=True)
        chat_container = st.container(height=380)
        with chat_container:
            chat_history = st.session_state.usecase_chats[sub]
            for msg in chat_history:
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    st.markdown(f'<div class="chat-bubble-pill-user">{content}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-row-horizontal">
                            <div class="chat-avatar-col">
                                <div class="chat-avatar-icon">{"🛡️" if is_sec else "❤️" if is_wel else "🏙️"}</div>
                                <div class="chat-avatar-label">{"SEC" if is_sec else "WEL" if is_wel else "CIV"}</div>
                            </div>
                            <div class="chat-bubble-col">
                                <div class="chat-bubble-pill-bot">{content}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top: 15px; margin-bottom: 10px;">', unsafe_allow_html=True)
        if sub == "sec_sleeper":
            all_nodes = db.get_all_nodes()
            pids = sorted([n.person_id for n in all_nodes]) if all_nodes else ["42"]
            selected_pid = st.selectbox("Select Target Person ID for sleeper cell scan", pids, index=pids.index("42") if "42" in pids else 0)
            st.session_state.selected_pid_val = selected_pid
            cols_sugg = st.columns(2)
            with cols_sugg[0]:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button("Scan for Sleeper Cell Group", key="sugg_sec_1"):
                    submit_security_query(f"Scan for Sleeper Cell Group containing P{selected_pid}", db)
                st.markdown('</div>', unsafe_allow_html=True)
            with cols_sugg[1]:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button("Evaluate louvain partition", key="sugg_sec_2"):
                    submit_security_query(f"Evaluate louvain partition for P{selected_pid}", db)
                st.markdown('</div>', unsafe_allow_html=True)
        elif sub == "sec_associates":
            all_nodes = db.get_all_nodes()
            pids = sorted([n.person_id for n in all_nodes]) if all_nodes else ["42"]
            selected_pid = st.selectbox("Select Target Person ID to list connections", pids, index=pids.index("42") if "42" in pids else 0)
            st.session_state.selected_pid_val = selected_pid
            cols_sugg = st.columns(2)
            with cols_sugg[0]:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button("Identify accomplices", key="sugg_sec_3"):
                    submit_security_query(f"Identify accomplices of P{selected_pid}", db)
                st.markdown('</div>', unsafe_allow_html=True)
            with cols_sugg[1]:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button("Check co-presence history", key="sugg_sec_4"):
                    submit_security_query(f"Check co-presence history of P{selected_pid}", db)
                st.markdown('</div>', unsafe_allow_html=True)
        elif sub == "wel_missing":
            all_nodes = db.get_all_nodes()
            pids = sorted([n.person_id for n in all_nodes]) if all_nodes else ["19"]
            selected_pid = st.selectbox("Select Target Person ID for search query", pids, index=pids.index("19") if "19" in pids else 0)
            st.session_state.selected_pid_val = selected_pid
            cols_sugg = st.columns(2)
            with cols_sugg[0]:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button(f"Solve case for P{selected_pid}", key="sugg_wel_1"):
                    submit_welfare_query(f"Solve missing child case for P{selected_pid}", selected_pid, db)
                st.markdown('</div>', unsafe_allow_html=True)
            with cols_sugg[1]:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button(f"Predict next camera for P{selected_pid}", key="sugg_wel_2"):
                    submit_welfare_query(f"Predict next camera location for P{selected_pid}", selected_pid, db)
                st.markdown('</div>', unsafe_allow_html=True)
        elif sub == "wel_density":
            selected_cam = st.selectbox("Select Camera for Density Check", ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7"], index=0)
            st.session_state.selected_cam_val = selected_cam
            cols_sugg = st.columns(2)
            with cols_sugg[0]:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button(f"Check crowd density for {selected_cam}", key="sugg_wel_3"):
                    submit_welfare_query(f"Check crowd density for {selected_cam}", "19", db)
                st.markdown('</div>', unsafe_allow_html=True)
            with cols_sugg[1]:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button(f"Evaluate flow rates for {selected_cam}", key="sugg_wel_4"):
                    submit_welfare_query(f"Evaluate flow rates for {selected_cam}", "19", db)
                st.markdown('</div>', unsafe_allow_html=True)
        elif sub == "civ_traffic":
            cols_sugg = st.columns(2)
            with cols_sugg[0]:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button("Check helmet infractions", key="sugg_civ_1"):
                    submit_civic_query("Check helmet infractions")
                st.markdown('</div>', unsafe_allow_html=True)
            with cols_sugg[1]:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button("Scan for triple riding", key="sugg_civ_2"):
                    submit_civic_query("Scan for triple riding")
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        query = st.chat_input("Ask Agent...", key="agent_chat_input")
        if query:
            if is_sec:
                submit_security_query(query, db)
            elif is_wel:
                pid_to_pass = st.session_state.get("selected_pid_val", "19")
                submit_welfare_query(query, pid_to_pass, db)
            elif is_civ:
                submit_civic_query(query)

    with col_console_right:
        st.markdown('<div class="chat-visual-card">', unsafe_allow_html=True)
        
        has_payload = False
        if is_sec:
            if st.session_state.security_payload.get("status") == "SUCCESS":
                has_payload = True
                payload = st.session_state.security_payload
                st.markdown(f"""
                    <div style="padding:10px 14px; background:rgba(239, 68, 68, 0.08); border:1px solid rgba(239, 68, 68, 0.2); border-radius:10px; margin-bottom:10px;">
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#EF4444; font-weight:800;">ALERT: {payload['threat_classification']}</div>
                        <div style="font-size:0.78rem; color:#FFF; margin-top:4px;">Target Cell Node ID: <b>P{payload['target_id']}</b></div>
                    </div>
                """, unsafe_allow_html=True)
                relations = []
                graph_data = db.get_person_graph(payload['target_id'])
                if graph_data:
                    for conn in graph_data.get("connections", []):
                        relations.append({
                            "person_id": conn["person_id"],
                            "confidence": conn["confidence"],
                            "relationship": conn["relationship"],
                            "color": conn["color"]
                        })
                fig_sec = plot_security_network(payload['target_id'], relations[:6])
                if fig_sec:
                    st.plotly_chart(fig_sec, width="stretch")
        
        elif is_wel:
            if st.session_state.welfare_payload.get("status") == "SUCCESS":
                has_payload = True
                payload = st.session_state.welfare_payload
                if "predictions" in payload:
                    target_id = payload.get("target_id", "19")
                    fig_wel = plot_welfare_path(payload, target_id)
                    st.plotly_chart(fig_wel, width="stretch")
                elif "metrics" in payload:
                    fig_crowd = plot_crowd_density(payload)
                    st.plotly_chart(fig_crowd, width="stretch")
        
        elif is_civ:
            has_payload = True
            st.markdown(render_violations_table(), unsafe_allow_html=True)

        if not has_payload:
            st.markdown('<div style="text-align:center; color:#64748b; font-size:0.8rem; font-family:\'JetBrains Mono\', monospace; padding:100px 0;">INTELLIGENCE VISUAL SCREEN<br>[Initialize query to map telemetry]</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def plot_security_network(target_id, connections):
    fig = go.Figure()
    N = len(connections)
    if N == 0:
        return None
    for i, conn in enumerate(connections):
        angle = 2 * np.pi * i / N
        x_pos = 4 * np.cos(angle)
        y_pos = 4 * np.sin(angle)
        fig.add_trace(go.Scatter(
            x=[0, x_pos], y=[0, y_pos],
            mode="lines",
            line=dict(color=conn.get("color", "#4DA8FF"), width=conn.get("confidence", 0.5) * 5),
            hoverinfo="none",
            showlegend=False
        ))
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    for i, conn in enumerate(connections):
        angle = 2 * np.pi * i / N
        node_x.append(4 * np.cos(angle))
        node_y.append(4 * np.sin(angle))
        node_text.append(f"<b>P{conn['person_id']}</b><br>Confidence: {conn['confidence']:.2f}<br>Type: {conn['relationship']}")
        node_colors.append(conn.get("color", "#4DA8FF"))
        node_sizes.append(15 + conn.get("confidence", 0.5) * 15)
    node_x.append(0)
    node_y.append(0)
    node_text.append(f"<b>TARGET: P{target_id}</b><br>Threat Group Source")
    node_colors.append("#EF4444")
    node_sizes.append(28)
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color="white")),
        text=[t.split("<b>")[1].split("</b>")[0] for t in node_text],
        textposition="top center",
        textfont=dict(color="#AAB6C5", size=9, family="JetBrains Mono"),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        hoverlabel=dict(bgcolor="#0B1622", bordercolor="rgba(255,255,255,0.08)", font=dict(color="#F8FAFC", family="Inter"))
    )
    return fig

def plot_welfare_path(payload, target_id):
    fig_track = go.Figure()
    corridors = [
        ([0, 10], [2, 2]), ([0, 10], [5, 5]), ([0, 10], [8, 8]),
        ([2, 2], [0, 10]), ([5, 5], [0, 10]), ([8, 8], [0, 10])
    ]
    for x_c, y_c in corridors:
        fig_track.add_trace(go.Scatter(
            x=x_c, y=y_c, mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.04)", width=1),
            showlegend=False, hoverinfo="skip"
        ))
    preds = payload.get("predictions", {})
    cx, cy = preds.get("last_seen", (5.0, 5.0))
    future_path = preds.get("projected_path_trail", [])
    theta = np.linspace(0, 2*np.pi, 100)
    r_scan = 1.8
    cx_circle = cx + r_scan * np.cos(theta)
    cy_circle = cy + r_scan * np.sin(theta)
    fig_track.add_trace(go.Scatter(
        x=cx_circle, y=cy_circle, mode="lines",
        line=dict(color="rgba(6, 182, 212, 0.25)", width=1, dash="dash"),
        name="Perimeter Sweep", fill="toself",
        fillcolor="rgba(6, 182, 212, 0.02)", hoverinfo="skip"
    ))
    fig_track.add_trace(go.Scatter(
        x=[cx], y=[cy], mode="markers+text",
        marker=dict(size=14, color="#06B6D4", line=dict(width=2, color="white"), symbol="circle-dot"),
        text=[f"LAST KNOWN // P{target_id}"], textposition="top right",
        textfont=dict(color="#06B6D4", family="JetBrains Mono", size=9),
        name="Target Last Seen"
    ))
    if future_path:
        fx, fy = zip(*future_path)
        fig_track.add_trace(go.Scatter(
            x=fx, y=fy, mode="lines+markers",
            line=dict(dash="dot", color="#F59E0B", width=2.5),
            marker=dict(size=7, color="#F59E0B", symbol="triangle-up"),
            name="Projected Trail"
        ))
    fig_track.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False, tickfont=dict(color="#64748b", family="JetBrains Mono", size=8)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False, tickfont=dict(color="#64748b", family="JetBrains Mono", size=8)),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        legend=dict(font=dict(color="#AAB6C5", size=8, family="Inter"), bgcolor="rgba(0,0,0,0)", yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig_track

def plot_crowd_density(payload):
    curr_count = payload.get("metrics", {}).get("occupancy_density_count", 15.0)
    grid = np.random.rand(20, 20) * 1.5
    if curr_count > 0:
        grid = grid + (curr_count * 0.4)
    fig_heat = go.Figure(data=[go.Surface(
        z=grid, colorscale="Viridis",
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="#fff", project_z=True))
    )])
    fig_heat.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(gridcolor="rgba(255,255,255,0.03)", backgroundcolor="rgba(0,0,0,0)", showbackground=False, tickfont=dict(size=7)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.03)", backgroundcolor="rgba(0,0,0,0)", showbackground=False, tickfont=dict(size=7)),
            zaxis=dict(gridcolor="rgba(255,255,255,0.03)", backgroundcolor="rgba(0,0,0,0)", showbackground=False, tickfont=dict(size=7))
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280
    )
    return fig_heat

def render_violations_table():
    violations_data = [
        {"id": "V-982", "cam": "cam3", "infraction": "Triple Riding (Motorcycle)", "pid": "P25", "status": "Fine Issued", "action": "Behavior Pattern Linked"},
        {"id": "V-981", "cam": "cam1", "infraction": "No Helmet Detected", "pid": "P12", "status": "Fine Issued", "action": "Warning Logged"},
        {"id": "V-980", "cam": "cam5", "infraction": "Littering (Civic Violation)", "pid": "P36", "status": "Flagged", "action": "Social Compliance Warning"},
        {"id": "V-979", "cam": "cam2", "infraction": "Public Nuisance", "pid": "P9", "status": "Dispatched", "action": "Patrol Officer Notified"},
    ]
    rows_html = ""
    for v in violations_data:
        status_class = "status-fine" if v["status"] == "Fine Issued" else "status-flagged" if v["status"] == "Flagged" else "status-dispatched"
        rows_html += f"""
        <tr>
            <td style="font-family:'JetBrains Mono', monospace; font-weight:700; color:#F8FAFC;">{v['id']}</td>
            <td style="color:#AAB6C5;">{v['cam']}</td>
            <td style="color:#E2E8F0; font-weight:500;">{v['infraction']}</td>
            <td style="font-family:'JetBrains Mono', monospace; color:#7CFFB2;">{v['pid']}</td>
            <td><span class="status-pill {status_class}">{v['status']}</span></td>
            <td style="font-size:0.75rem; color:#AAB6C5;">{v['action']}</td>
        </tr>
        """
    table_html = f"""
    <div style="overflow-x:auto;">
        <table class="civic-table">
            <thead>
                <tr>
                    <th>VIOLATION ID</th>
                    <th>CAMERA</th>
                    <th>INFRACTION</th>
                    <th>PERSON ID</th>
                    <th>STATUS</th>
                    <th>ACTION TAKEN</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    return table_html

def submit_security_query(query: str, db):
    sub = st.session_state.active_subcase
    st.session_state.usecase_chats[sub].append({"role": "user", "content": query})
    from usecases.threat import ThreatNetworkPredictiveAgent
    from Agents.anomaly import BehavioralAnomalyAgent
    anomaly_agent = BehavioralAnomalyAgent(anomaly_sensitivity_threshold=3.0)
    import re
    match = re.search(r'(?:P|pedestrian|person)?\s*(\d+)', query, re.IGNORECASE)
    target_id = match.group(1) if match else "42"
    q_low = query.lower()
    is_explicit = False
    if sub == "sec_sleeper":
        is_explicit = any(x in q_low for x in ["scan", "partition", "louvain", "group", "cell"])
    elif sub == "sec_associates":
        is_explicit = any(x in q_low for x in ["accomplice", "co-presence", "history", "associate", "identify", "connections", "linkage"])
    if not is_explicit:
        try:
            from agent import natural_language_query
            nodes_raw = db.get_all_nodes()
            edges_raw = db.get_all_edges()
            deg_map = {}
            for e in edges_raw:
                if e.relationship != "stranger":
                    deg_map[e.person_id_a] = deg_map.get(e.person_id_a, 0) + 1
                    deg_map[e.person_id_b] = deg_map.get(e.person_id_b, 0) + 1
            gd = {
                "nodes": [{"id": n.person_id, "degree": deg_map.get(n.person_id, 0), "last_seen": getattr(n, "last_seen", None)} for n in nodes_raw],
                "edges": [{"person_id_a": e.person_id_a, "person_id_b": e.person_id_b, "confidence": e.confidence, "relationship": e.relationship, "total_meetings": e.total_meetings} for e in edges_raw],
            }
            reply = natural_language_query(query, gd)
            st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
            st.rerun()
            return
        except Exception:
            pass
    if sub == "sec_sleeper":
        if any(x in q_low for x in ["accomplice", "associate", "connection", "relationship", "contact"]):
            reply = (
                "⚠️ **Security Agent Warning:**\n\n"
                "This agent is dedicated to Sleeper Cell Patterns. "
                "To search for connections, accomplices, and relationship mapping, switch to the Hidden Associates agent."
            )
            st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
            st.rerun()
            return
        comms = db.get_louvain_communities()
        target_cell = -1
        cell_members = []
        for c in comms:
            m_strs = [str(node) for node in c["members"]]
            if target_id in m_strs:
                target_cell = c["community_id"]
                cell_members = [f"P{m}" for m in c["members"]]
                break
        if target_cell != -1:
            reply = (
                f"🛡️ **Security Agent Louvain Partition Scan:**\n\n"
                f"Target suspect **P{target_id}** is assigned to **Sleeper Cell Group #{target_cell}**.\n\n"
                f"**Discovered Cell Nodes:** {', '.join(cell_members)}.\n"
                f"Operational Warning: Nodes in this group have high contact density rates."
            )
        else:
            reply = (
                f"🛡️ **Security Agent Louvain Partition Scan:**\n\n"
                f"Target suspect **P{target_id}** is not active in any coordinated sleeper cell groups."
            )
        t_base = time.time()
        mock_history = [(5.0 + np.sin(i*1.5)*0.8, 4.0 + np.cos(i*1.5)*0.8, t_base + i*0.04) for i in range(16)]
        anomaly_res = anomaly_agent.evaluate_entity_tracklet(target_id, mock_history)
        clusters = {target_cell: [m.replace("P","") for m in cell_members]} if target_cell != -1 else {}
        relations = []
        graph_data = db.get_person_graph(target_id)
        if graph_data:
            for conn in graph_data.get("connections", []):
                relations.append({
                    "entity_id": conn["person_id"],
                    "confidence_score": conn["confidence"],
                    "total_duration_sec": conn.get("duration", 60.0)
                })
        agent = ThreatNetworkPredictiveAgent(behavioral_threat_threshold=3.0)
        res = agent.execute_threat_network_discovery(target_id, anomaly_res, clusters, relations)
        st.session_state.security_payload = res
        st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
        st.rerun()
    elif sub == "sec_associates":
        if any(x in q_low for x in ["sleeper", "cell", "group", "community", "partition", "cluster"]):
            reply = (
                "⚠️ **Security Agent Warning:**\n\n"
                "This agent is dedicated to Hidden Associates detection. "
                "To perform social partitioning and group coordinate density analysis, switch to the Sleeper Cell Patterns agent."
            )
            st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
            st.rerun()
            return
        relations_list = []
        try:
            graph_data = db.get_person_graph(target_id)
            if graph_data:
                for conn in graph_data.get("connections", []):
                    relations_list.append(f"P{conn['person_id']} ({conn['relationship'].replace('_',' ').title()} - {int(conn['confidence']*100)}% conf)")
        except Exception:
            pass
        if relations_list:
            reply = (
                f"⚠️ **Security Agent High-Risk Proximity Leads:**\n\n"
                f"Suspect **P{target_id}** has direct threat linkages with:\n"
                f"- " + "\n- ".join(relations_list[:5]) + "\n\n"
                f"Isolate accomplice coordinates to prevent group mobilization."
            )
        else:
            reply = (
                f"⚠️ **Security Agent Proximity Leads:**\n\n"
                f"No direct high-risk connections logged for suspect **P{target_id}**."
            )
        t_base = time.time()
        mock_history = [(5.0 + np.sin(i*1.5)*0.8, 4.0 + np.cos(i*1.5)*0.8, t_base + i*0.04) for i in range(16)]
        anomaly_res = anomaly_agent.evaluate_entity_tracklet(target_id, mock_history)
        relations = []
        graph_data = db.get_person_graph(target_id)
        if graph_data:
            for conn in graph_data.get("connections", []):
                relations.append({
                    "entity_id": conn["person_id"],
                    "confidence_score": conn["confidence"],
                    "total_duration_sec": conn.get("duration", 60.0)
                })
        agent = ThreatNetworkPredictiveAgent(behavioral_threat_threshold=3.0)
        res = agent.execute_threat_network_discovery(target_id, anomaly_res, {}, relations)
        st.session_state.security_payload = res
        st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
        st.rerun()

def submit_welfare_query(query: str, selected_pid: str, db):
    sub = st.session_state.active_subcase
    st.session_state.usecase_chats[sub].append({"role": "user", "content": query})
    import re
    match = re.search(r'(?:P|pedestrian|person)?\s*(\d+)', query, re.IGNORECASE)
    pid = match.group(1) if match else selected_pid
    q_low = query.lower()
    if sub == "wel_missing":
        if any(x in q_low for x in ["crowd", "density", "choke", "safety", "stampede", "hotspots", "flow rate"]):
            reply = (
                "⚠️ **Welfare Agent Warning:**\n\n"
                "This agent is dedicated exclusively to **Missing Child Tracing**. "
                "For crowd density metrics, choke points, or stampede prevention, please switch to the **Crowd Density** agent."
            )
            st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
            st.rerun()
            return
    elif sub == "wel_density":
        if any(x in q_low for x in ["missing", "child", "who", "seen with", "accomplice", "associate", "connection", "friend", "camera", "cam", "location", "last seen"]):
            reply = (
                "⚠️ **Welfare Agent Warning:**\n\n"
                "This agent is dedicated exclusively to **Crowd Density & Safety**. "
                "For missing person coordinate tracking, camera history, or contact analysis, please switch to the **Missing Child Tracing** agent."
            )
            st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
            st.rerun()
            return
    is_explicit = False
    if sub == "wel_missing":
        is_explicit = any(x in q_low for x in ["camera", "cam", "location", "last seen", "seen", "who", "seen with", "accomplice", "associate", "connection", "friend", "solve", "investigate", "trajectory", "forecast", "recovery", "pipeline", "interception"])
    elif sub == "wel_density":
        is_explicit = any(x in q_low for x in ["choke", "flow", "density", "capacity", "rate", "occupancy", "hazard", "bottleneck", "check"])
    if not is_explicit:
        try:
            from agent import natural_language_query
            nodes_raw = db.get_all_nodes()
            edges_raw = db.get_all_edges()
            deg_map = {}
            for e in edges_raw:
                if e.relationship != "stranger":
                    deg_map[e.person_id_a] = deg_map.get(e.person_id_a, 0) + 1
                    deg_map[e.person_id_b] = deg_map.get(e.person_id_b, 0) + 1
            gd = {
                "nodes": [{"id": n.person_id, "degree": deg_map.get(n.person_id, 0), "last_seen": getattr(n, "last_seen", None)} for n in nodes_raw],
                "edges": [{"person_id_a": e.person_id_a, "person_id_b": e.person_id_b, "confidence": e.confidence, "relationship": e.relationship, "total_meetings": e.total_meetings} for e in edges_raw],
            }
            reply = natural_language_query(query, gd)
            st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
            st.rerun()
            return
        except Exception:
            pass
    if sub == "wel_missing":
        if any(x in q_low for x in ["camera", "cam", "location", "last seen", "seen"]):
            camera_ids = []
            try:
                nd = db._graph.nodes[pid]
                camera_ids = list(nd.get("camera_ids", []))
            except Exception:
                pass
            if not camera_ids:
                camera_ids = ["cam1", "cam2"]
            next_cam = camera_ids[0] if camera_ids else "cam1"
            reply = (
                f"🕵️‍♂️ **Welfare Agent Camera Prediction:**\n\n"
                f"Target Person **P{pid}** was previously tracked across camera feed(s): **{', '.join(camera_ids)}**.\n\n"
                f"**Transition Vector Forecast:** Based on temporal movement patterns, the subject is projected to appear next on camera **{next_cam}**. Monitor that feed closely."
            )
            move_agent, anomaly_agent = load_predictive_agents()
            mock_history = [(1.0 + i*0.1, 2.0 + i*0.1, time.time() - (16-i)*0.1) for i in range(16)]
            from usecases.missing import MissingChildPredictiveAgent
            local_agent = MissingChildPredictiveAgent(move_agent, anomaly_agent, db)
            payload = local_agent.execute_investigative_pipeline(pid, mock_history)
            st.session_state.welfare_payload = payload
            st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
            st.rerun()
        elif any(x in q_low for x in ["who", "seen with", "accomplice", "associate", "connection", "friend"]):
            connections_list = []
            try:
                graph_data = db.get_person_graph(pid)
                if graph_data:
                    for conn in graph_data.get("connections", []):
                        connections_list.append(f"P{conn['person_id']} ({conn['relationship'].replace('_',' ').title()} - {int(conn['confidence']*100)}% conf)")
            except Exception:
                pass
            if connections_list:
                reply = (
                    f"🕸️ **Welfare Agent Relational Proximity Scan:**\n\n"
                    f"Target Person **P{pid}** has historical contacts and shared camera occurrences with:\n"
                    f"- " + "\n- ".join(connections_list[:5]) + "\n\n"
                    f"Isolate these accomplice profiles to determine if they are moving in coordination."
                )
            else:
                reply = (
                    f"🕸️ **Welfare Agent Relational Proximity Scan:**\n\n"
                    f"No close proximity connections or co-presence events registered in GraphDB for target **P{pid}**."
                )
            move_agent, anomaly_agent = load_predictive_agents()
            mock_history = [(1.0 + i*0.1, 2.0 + i*0.1, time.time() - (16-i)*0.1) for i in range(16)]
            from usecases.missing import MissingChildPredictiveAgent
            local_agent = MissingChildPredictiveAgent(move_agent, anomaly_agent, db)
            payload = local_agent.execute_investigative_pipeline(pid, mock_history)
            st.session_state.welfare_payload = payload
            st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
            st.rerun()
        else:
            api_success = False
            payload = {}
            try:
                res = requests.get(f"http://127.0.0.1:8000/api/v1/usecases/missing/{pid}", timeout=0.5)
                if res.status_code == 200:
                    payload = res.json()
                    if payload.get("status") == "SUCCESS":
                        api_success = True
            except Exception:
                pass
            if not api_success:
                move_agent, anomaly_agent = load_predictive_agents()
                mock_history = [
                    (0.5, 1.2, 1779959333.54), (0.7, 1.5, 1779959333.57),
                    (0.9, 1.8, 1779959333.61), (1.1, 2.1, 1779959333.64),
                    (1.3, 2.4, 1779959333.67), (1.5, 2.7, 1779959333.71),
                    (1.7, 3.0, 1779959333.74), (1.9, 3.3, 1779959333.77),
                    (2.1, 3.6, 1779959333.80), (2.3, 3.9, 1779959333.84),
                    (2.5, 4.2, 1779959333.87), (2.7, 4.5, 1779959333.90),
                    (2.9, 4.8, 1779959334.00), (3.2, 5.1, 1779959334.04),
                    (3.7, 5.4, 1779959334.08), (4.2, 5.7, 1779959334.12)
                ]
                from usecases.missing import MissingChildPredictiveAgent
                local_agent = MissingChildPredictiveAgent(move_agent, anomaly_agent, db)
                payload = local_agent.execute_investigative_pipeline(pid, mock_history)
            st.session_state.welfare_payload = payload
            st.session_state.usecase_chats[sub].append({"role": "assistant", "content": payload["chatbot_payload"]})
            st.rerun()
    elif sub == "wel_density":
        cam_match = re.search(r"(cam\d+)", q_low)
        cam = cam_match.group(1) if cam_match else st.session_state.get("selected_cam_val", "cam1")
        api_success = False
        payload = {}
        try:
            res = requests.get(f"http://127.0.0.1:8000/api/v1/usecases/crowd/{cam}", timeout=0.5)
            if res.status_code == 200:
                payload = res.json()
                if payload.get("status") == "SUCCESS":
                    api_success = True
        except Exception:
            pass
        if not api_success:
            from usecases.crowd import CrowdSafetyPredictiveAgent
            local_crowd_agent = CrowdSafetyPredictiveAgent(surge_velocity_threshold=1.5, structural_capacity=30)
            curr_count = float(PS.G.get("person_count", 0))
            if curr_count <= 0:
                curr_count = 14.0
            mock_hist = [(max(0.0, curr_count - 1.0), time.time() - 1.0), (curr_count, time.time())]
            sim_map = np.random.rand(20, 20) * 0.1
            if curr_count > 0:
                sim_map = sim_map + (curr_count * 0.05)
            payload = local_crowd_agent.evaluate_choke_point_metrics(cam, curr_count, mock_hist, sim_map)
        st.session_state.welfare_payload = payload
        st.session_state.usecase_chats[sub].append({"role": "assistant", "content": payload["chatbot_payload"]})
        st.rerun()

def submit_civic_query(query: str):
    sub = st.session_state.active_subcase
    st.session_state.usecase_chats[sub].append({"role": "user", "content": query})
    reply = ""
    q_low = query.lower()
    is_civic_cmd = any(x in q_low for x in ["helmet", "infraction", "triple", "traffic", "civic", "compliance", "littering", "nuisance"])
    if not is_civic_cmd:
        try:
            from agent import natural_language_query
            db = PS.G.get("graph_db")
            if db is None:
                from usecases_page import load_graph_database
                db = load_graph_database()
            nodes_raw = db.get_all_nodes()
            edges_raw = db.get_all_edges()
            deg_map = {}
            for e in edges_raw:
                if e.relationship != "stranger":
                    deg_map[e.person_id_a] = deg_map.get(e.person_id_a, 0) + 1
                    deg_map[e.person_id_b] = deg_map.get(e.person_id_b, 0) + 1
            gd = {
                "nodes": [{"id": n.person_id, "degree": deg_map.get(n.person_id, 0), "last_seen": getattr(n, "last_seen", None)} for n in nodes_raw],
                "edges": [{"person_id_a": e.person_id_a, "person_id_b": e.person_id_b, "confidence": e.confidence, "relationship": e.relationship, "total_meetings": e.total_meetings} for e in edges_raw],
            }
            reply = natural_language_query(query, gd)
            st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
            st.rerun()
            return
        except Exception:
            pass
    from usecases.civic import CivicOrderPredictiveAgent
    import requests
    local_agent = CivicOrderPredictiveAgent(loitering_time_threshold_sec=10.0)

    if "helmet" in q_low or "infraction" in q_low or "triple" in q_low or "traffic" in q_low:
        api_success = False
        payload = {}
        try:
            res = requests.get("http://127.0.0.1:8000/api/v1/usecases/civic/traffic/MOTORCYCLE_NODE_77", timeout=0.5)
            if res.status_code == 200:
                payload = res.json()
                if payload.get("status") == "SUCCESS":
                    api_success = True
        except Exception:
            pass
        if not api_success:
            payload = local_agent.process_traffic_compliance(
                vehicle_id="MOTORCYCLE_NODE_77",
                associated_pedestrian_ids=["WILDTRACK_PED_01", "WILDTRACK_PED_02", "WILDTRACK_PED_03"],
                helmet_detected_flags={"WILDTRACK_PED_01": True, "WILDTRACK_PED_02": False, "WILDTRACK_PED_03": True}
            )
        violations_str = ", ".join([v["type"] for v in payload["active_violations"]])
        reply = (
            f"🚨 **URG-IS CIVIC AUTOMATION REPORT: TRAFFIC COMPLIANCE** 🚨\n\n"
            f"**Vehicle Identifier:** `{payload['vehicle_identifier']}`\n"
            f"**Compliance Status:** `{payload['compliance_status']}`\n"
            f"**Active Violations:** {violations_str if violations_str else 'None'}\n\n"
            f"**Details:**\n" + "\n".join([f"- {v['details']}" for v in payload["active_violations"]]) + f"\n\n"
            f"*(Execution Latency: {payload['latency_ms']} ms)*"
        )
    elif "civic" in q_low or "compliance" in q_low or "littering" in q_low or "nuisance" in q_low:
        api_success = False
        payload = {}
        try:
            res = requests.get("http://127.0.0.1:8000/api/v1/usecases/civic/compliance/WILDTRACK_PED_42", timeout=0.5)
            if res.status_code == 200:
                payload = res.json()
                if payload.get("status") == "SUCCESS":
                    api_success = True
        except Exception:
            pass
        if not api_success:
            mock_active_loiter_history = [
                (2.1, 3.4, time.time() - 12.0),
                (2.2, 3.5, time.time())
            ]
            mock_abandoned_object_event = [
                {"event_type": "OBJECT_ABANDONED", "class_label": "trash", "coordinates": (2.15, 3.45)}
            ]
            payload = local_agent.process_civic_compliance(
                entity_id="WILDTRACK_PED_42",
                spatial_history_meters=mock_active_loiter_history,
                object_permanence_events=mock_abandoned_object_event
            )
        reply = payload["chatbot_payload"] + f"\n\n*(Execution Latency: {payload['latency_ms']} ms)*"
    else:
        reply = (
            "⚠️ **Smart City Agent Warning:**\n\n"
            "This agent is dedicated to Smart City Civic and Traffic Violations. "
            "Please query traffic violations, helmet gaps, triple-riding, or littering compliance."
        )
    st.session_state.usecase_chats[sub].append({"role": "assistant", "content": reply})
    st.rerun()
