import time
import logging
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, Query
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any

# Disable default Uvicorn access logging to keep the terminal silent
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from Agents.movement import ZeroShotMovementAgent
from Agents.crowd import CrowdDensitySurgeAgent
from Agents.anomaly import BehavioralAnomalyAgent
from usecases.missing import MissingChildPredictiveAgent
from usecases.crowd import CrowdSafetyPredictiveAgent
from usecases.threat import ThreatNetworkPredictiveAgent
from usecases.civic import CivicOrderPredictiveAgent
from core.graph.graph_db import GraphDB

app = FastAPI(title="URG-IS Core Agent Orchestrator")

# Core engine allocations (kept fully intact)
movement_agent = ZeroShotMovementAgent(prediction_seconds=5)
crowd_agent = CrowdDensitySurgeAgent(surge_threshold_per_sec=1.5)
anomaly_agent = BehavioralAnomalyAgent(anomaly_sensitivity_threshold=3.0)
db_conn = GraphDB(snapshot_path="data/snapshots/prod_graph.json")

missing_agent = MissingChildPredictiveAgent(
    movement_agent_instance=movement_agent,
    anomaly_agent_instance=anomaly_agent,
    graph_database_connection=db_conn
)
crowd_safety_agent = CrowdSafetyPredictiveAgent(surge_velocity_threshold=1.5, structural_capacity=30)
threat_network_agent = ThreatNetworkPredictiveAgent(behavioral_threat_threshold=3.0)
civic_order_agent = CivicOrderPredictiveAgent(loitering_time_threshold_sec=10.0)

TRACKING_RAM_BUFFER = {}
CAMERA_ACTIVE_PEOPLE = {}
CAMERA_COUNT_HISTORY = {}

class TelemetryPacket(BaseModel):
    camera_id: str
    entity_id: str
    x_meters: float
    y_meters: float
    frame_timestamp: float

# ─── SILENT BACKEND DATA PIPELINES (ALL PRINTS REMOVED) ───────────────────────
@app.post("/api/v1/telemetry")
async def ingest_tracking_telemetry(packet: TelemetryPacket):
    eid = packet.entity_id
    cid = packet.camera_id
    current_time = packet.frame_timestamp
    current_coords = (packet.x_meters, packet.y_meters)

    if eid not in TRACKING_RAM_BUFFER: TRACKING_RAM_BUFFER[eid] = []
    TRACKING_RAM_BUFFER[eid].append((current_coords[0], current_coords[1], current_time))
    if len(TRACKING_RAM_BUFFER[eid]) > 50: TRACKING_RAM_BUFFER[eid].pop(0)

    if cid not in CAMERA_ACTIVE_PEOPLE: CAMERA_ACTIVE_PEOPLE[cid] = {}
    CAMERA_ACTIVE_PEOPLE[cid][eid] = current_time
    
    cutoff = current_time - 2.0
    CAMERA_ACTIVE_PEOPLE[cid] = {e: ts for e, ts in CAMERA_ACTIVE_PEOPLE[cid].items() if ts > cutoff}
    current_count = len(CAMERA_ACTIVE_PEOPLE[cid])
    
    if cid not in CAMERA_COUNT_HISTORY: CAMERA_COUNT_HISTORY[cid] = []
    CAMERA_COUNT_HISTORY[cid].append((float(current_count), current_time))
    if len(CAMERA_COUNT_HISTORY[cid]) > 50: CAMERA_COUNT_HISTORY[cid].pop(0)
    
    asyncio.create_task(run_background_agent_analysis(eid, packet.camera_id))
    return {"status": "QUEUED"}

async def run_background_agent_analysis(entity_id: str, camera_id: str):
    history = TRACKING_RAM_BUFFER[entity_id]
    pure_coords = [(pt[0], pt[1]) for pt in history]
    if len(pure_coords) >= 8:
        await asyncio.get_event_loop().run_in_executor(None, movement_agent.forecast_track, entity_id, pure_coords)
    if len(history) >= 16:
        anomaly_agent.evaluate_entity_tracklet(entity_id, history)

@app.websocket("/ws/dashboard/alerts")
async def dashboard_websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True: await asyncio.sleep(1)
    except Exception: pass


# ─── USE CASE 1: MISSING PERSONS INTERACTION TRACKER ─────────────────────────
@app.get("/api/v1/usecases/missing/{entity_id}")
async def get_missing_child_recovery(entity_id: str, q: str = ""):
    # Print clean diagnostics only when an analyst requests it
    print("\n" + "🔍" * 40)
    print(f"📥 [FRONTEND DATA INPUT] -> TARGET WORKSPACE DESK: WEL // MISSING")
    print(f"   ↳ Subject Person Node ID : P{entity_id}")
    print(f"   ↳ Analyst Input Query    : \"{q if q else 'Default Kinematic Sweep Command'}\"")
    print("" * 80)

    if entity_id not in TRACKING_RAM_BUFFER or len(TRACKING_RAM_BUFFER[entity_id]) < 8:
        base_ts = time.time() - 8.0
        TRACKING_RAM_BUFFER[entity_id] = [(2.0 + i * 0.15, 3.0 + i * 0.1, base_ts + i * 0.5) for i in range(16)]

    history = TRACKING_RAM_BUFFER[entity_id]
    
    # Process text using our working Gemma module definitions
    loop = asyncio.get_event_loop()
    response_payload = await loop.run_in_executor(None, missing_agent.execute_investigative_pipeline, entity_id, history, q)

    print(f"📤 [BACKEND RESPONSE PACKET OUTBOUND]")
    print(f"   ↳ System Status : {response_payload.get('status')}")
    print(f"   ↳ Model Cognitive Briefing Text:\n")
    print(response_payload.get("chatbot_payload", "").strip())
    print("🔍" * 40 + "\n")
    return response_payload


# ─── USE CASE 2: CROWD SAFETY MANAGEMENT INTERACTION TRACKER ──────────────────
@app.get("/api/v1/usecases/crowd/{camera_id}")
async def get_crowd_safety_analysis(camera_id: str, q: str = ""):
    # Print clean diagnostics only when an analyst requests it
    print("\n" + "📊" * 40)
    print(f"📥 [FRONTEND DATA INPUT] -> TARGET WORKSPACE DESK: WEL // DENSITY")
    print(f"   ↳ Target Camera Sensor   : {camera_id.upper()}")
    print(f"   ↳ Analyst Input Query    : \"{q if q else 'Default Volumetric Flow Command'}\"")
    print("" * 80)

    current_count = len(CAMERA_ACTIVE_PEOPLE.get(camera_id, {}))
    if current_count == 0:
        current_count = int(db_conn.get_all_nodes() and len(db_conn.get_all_nodes()) or 14)

    history = CAMERA_COUNT_HISTORY.get(camera_id, [])
    if not history:
        history = [(max(0.0, current_count - 1.0), time.time() - 5.0), (current_count, time.time())]

    simulated_density = np.random.rand(20, 20) * 0.1 + (float(current_count) * 0.05)
    
    loop = asyncio.get_event_loop()
    response_payload = await loop.run_in_executor(
        None, crowd_safety_agent.evaluate_choke_point_metrics, camera_id, float(current_count), history, simulated_density, q
    )

    if "metrics" in response_payload:
        response_payload["metrics"]["occupancy_density_count"] = float(response_payload["metrics"]["occupancy_density_count"])
        response_payload["metrics"]["surge_velocity_rate"] = float(response_payload["metrics"]["surge_velocity_rate"])

    print(f"📤 [BACKEND RESPONSE PACKET OUTBOUND]")
    print(f"   ↳ System Status : {response_payload.get('status')}")
    print(f"   ↳ Model Cognitive Briefing Text:\n")
    print(response_payload.get("chatbot_payload", "").strip())
    print("📊" * 40 + "\n")
    return response_payload


# ─── ADDITIONAL SILENT CHANNELS ──────────────────────────────────────────────
@app.get("/api/v1/usecases/civic/traffic/{vehicle_id}")
async def get_traffic_compliance(vehicle_id: str):
    return civic_order_agent.process_traffic_compliance(vehicle_id, ["W_PED_01"], {"W_PED_01": True})

@app.get("/api/v1/usecases/civic/compliance/{entity_id}")
async def get_civic_compliance(entity_id: str):
    return civic_order_agent.process_civic_compliance(entity_id, [(2.1, 3.4, time.time())], [])

if __name__ == "__main__":
    import uvicorn
    # Enforce standard parsing arguments but drop system logging levels completely
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")