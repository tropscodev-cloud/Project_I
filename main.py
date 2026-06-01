import time
import asyncio
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from typing import List, Tuple, Dict

from Agents.movement import ZeroShotMovementAgent
from Agents.crowd import CrowdDensitySurgeAgent
from Agents.anomaly import BehavioralAnomalyAgent
from usecases.missing import MissingChildPredictiveAgent
from usecases.crowd import CrowdSafetyPredictiveAgent
from usecases.civic import CivicOrderPredictiveAgent
from core.graph.graph_db import GraphDB

app = FastAPI(title="URG-IS Core Agent Orchestrator")

movement_agent = ZeroShotMovementAgent(prediction_seconds=5)
crowd_agent = CrowdDensitySurgeAgent(surge_threshold_per_sec=1.5)
anomaly_agent = BehavioralAnomalyAgent(anomaly_sensitivity_threshold=3.0)
db_conn = GraphDB(snapshot_path="data/snapshots/prod_graph.json")
missing_agent = MissingChildPredictiveAgent(
    movement_agent_instance=movement_agent,
    anomaly_agent_instance=anomaly_agent,
    graph_database_connection=db_conn
)
crowd_safety_agent = CrowdSafetyPredictiveAgent(
    surge_velocity_threshold=1.5,
    structural_capacity=30
)
civic_order_agent = CivicOrderPredictiveAgent(
    loitering_time_threshold_sec=10.0
)

TRACKING_RAM_BUFFER = {}
CAMERA_ACTIVE_PEOPLE = {}
CAMERA_COUNT_HISTORY = {}

class TelemetryPacket(BaseModel):
    camera_id: str
    entity_id: str
    x_meters: float
    y_meters: float
    frame_timestamp: float

@app.post("/api/v1/telemetry")
async def ingest_tracking_telemetry(packet: TelemetryPacket):
    eid = packet.entity_id
    cid = packet.camera_id
    current_time = packet.frame_timestamp
    current_coords = (packet.x_meters, packet.y_meters)

    if eid not in TRACKING_RAM_BUFFER:
        TRACKING_RAM_BUFFER[eid] = []
    
    TRACKING_RAM_BUFFER[eid].append((current_coords[0], current_coords[1], current_time))
    
    if len(TRACKING_RAM_BUFFER[eid]) > 50:
        TRACKING_RAM_BUFFER[eid].pop(0)

    if cid not in CAMERA_ACTIVE_PEOPLE:
        CAMERA_ACTIVE_PEOPLE[cid] = {}
    CAMERA_ACTIVE_PEOPLE[cid][eid] = current_time
    
    cutoff = current_time - 2.0
    CAMERA_ACTIVE_PEOPLE[cid] = {
        e: ts for e, ts in CAMERA_ACTIVE_PEOPLE[cid].items() if ts > cutoff
    }
    current_count = len(CAMERA_ACTIVE_PEOPLE[cid])
    
    if cid not in CAMERA_COUNT_HISTORY:
        CAMERA_COUNT_HISTORY[cid] = []
    CAMERA_COUNT_HISTORY[cid].append((float(current_count), current_time))
    if len(CAMERA_COUNT_HISTORY[cid]) > 50:
        CAMERA_COUNT_HISTORY[cid].pop(0)
    
    surge_res = crowd_agent.monitor_surge_rate(current_count, current_time)
    if surge_res["status"] == "CRITICAL_SURGE_ALERT":
        print(f"[ORCHESTRATOR ALERT] Critical Crowd Surge on camera {cid} | Surge Rate: {surge_res['surge_rate_per_sec']}/s")

    asyncio.create_task(run_background_agent_analysis(eid, packet.camera_id))

    return {"status": "QUEUED"}

async def run_background_agent_analysis(entity_id: str, camera_id: str):
    history = TRACKING_RAM_BUFFER[entity_id]
    pure_coords = [(pt[0], pt[1]) for pt in history]
    
    if len(pure_coords) >= 8:
        loop = asyncio.get_event_loop()
        move_res = await loop.run_in_executor(
            None, agent_trajectory_wrapper, entity_id, pure_coords
        )
        if move_res["status"] == "SUCCESS":
            print(f"[ORCHESTRATOR] Movement Agent -> Forecasted 5 steps for {entity_id}. Next step: {move_res['predicted_trajectory'][0]}")

    if len(history) >= 16:
        anomaly_res = anomaly_agent.evaluate_entity_tracklet(entity_id, history)
        print(f"[ORCHESTRATOR] Anomaly Engine -> Evaluated {entity_id} | Score: {anomaly_res['anomaly_score']} | Status: {anomaly_res['status']}")
        
        if anomaly_res["status"] == "BEHAVIORAL_ANOMALY_DETECTED":
            print(f"[ORCHESTRATOR ALERT] Critical Behavioral Anomaly on {entity_id} | Score: {anomaly_res['anomaly_score']}")

def agent_trajectory_wrapper(entity_id, pure_coords):
    return movement_agent.forecast_track(entity_id, pure_coords)

@app.websocket("/ws/dashboard/alerts")
async def dashboard_websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(1)
    except Exception:
        print("[WS] Dashboard client disconnected.")

@app.get("/api/v1/usecases/missing/{entity_id}")
async def get_missing_child_recovery(entity_id: str):
    if entity_id not in TRACKING_RAM_BUFFER:
        return {"status": "GATHERING_CONTEXT", "message": f"No telemetry history found for entity {entity_id}."}
    history = TRACKING_RAM_BUFFER[entity_id]
    return missing_agent.execute_investigative_pipeline(entity_id, history)

@app.get("/api/v1/usecases/crowd/{camera_id}")
async def get_crowd_safety_analysis(camera_id: str):
    if camera_id not in CAMERA_ACTIVE_PEOPLE:
        return {"status": "INITIALIZING_BUFFER", "camera_id": camera_id, "message": "No active telemetry for this camera."}
    
    current_count = len(CAMERA_ACTIVE_PEOPLE.get(camera_id, {}))
    history = CAMERA_COUNT_HISTORY.get(camera_id, [])
    
    import numpy as np
    simulated_density = np.random.rand(20, 20) * 0.1
    if current_count > 0:
        simulated_density = simulated_density + (current_count * 0.05)
        
    res = crowd_safety_agent.evaluate_choke_point_metrics(
        camera_id=camera_id,
        current_occupant_count=float(current_count),
        historical_counts_timeline=history,
        raw_csrnet_density_map=simulated_density
    )
    
    if "metrics" in res:
        res["metrics"]["occupancy_density_count"] = float(res["metrics"]["occupancy_density_count"])
        res["metrics"]["surge_velocity_rate"] = float(res["metrics"]["surge_velocity_rate"])
        
    return res

@app.get("/api/v1/usecases/civic/traffic/{vehicle_id}")
async def get_traffic_compliance(vehicle_id: str):
    associated_pedestrian_ids = ["WILDTRACK_PED_01", "WILDTRACK_PED_02", "WILDTRACK_PED_03"]
    helmet_detected_flags = {"WILDTRACK_PED_01": True, "WILDTRACK_PED_02": False, "WILDTRACK_PED_03": True}
    return civic_order_agent.process_traffic_compliance(vehicle_id, associated_pedestrian_ids, helmet_detected_flags)

@app.get("/api/v1/usecases/civic/compliance/{entity_id}")
async def get_civic_compliance(entity_id: str):
    history = TRACKING_RAM_BUFFER.get(entity_id, [])
    if len(history) < 2:
        import time
        history = [
            (2.1, 3.4, time.time() - 12.0),
            (2.2, 3.5, time.time())
        ]
    mock_abandoned_object_event = [
        {"event_type": "OBJECT_ABANDONED", "class_label": "trash", "coordinates": (2.15, 3.45)}
    ]
    return civic_order_agent.process_civic_compliance(entity_id, history, mock_abandoned_object_event)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)