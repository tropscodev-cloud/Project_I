# api_server.py

"""FastAPI gateway that wraps URG‑IS agents and use‑case logic.
The Streamlit front‑end can call these HTTP endpoints to retrieve
information from the various agents (movement, crowd, anomaly, missing,
 civic, etc.). This file separates the orchestration from the original
`main.py` and provides a clean API surface.
"""

import os
import sys
from typing import Any, Dict, List
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in PYTHONPATH for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Agent imports – same classes used elsewhere in the code base.
# ---------------------------------------------------------------------------
from Agents.movement import ZeroShotMovementAgent
from Agents.crowd import CrowdDensitySurgeAgent
from Agents.anomaly import BehavioralAnomalyAgent
from usecases.missing import MissingChildPredictiveAgent
from usecases.crowd import CrowdSafetyPredictiveAgent
from usecases.civic import CivicOrderPredictiveAgent
from core.graph.graph_db import GraphDB

# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------
app = FastAPI(title="URG‑IS FastAPI Gateway")

# Allow Streamlit (localhost:8501) to access the API – can be tightened later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with "http://localhost:8501" for stricter policy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global objects – instantiated once when the server starts.
# ---------------------------------------------------------------------------
movement_agent = ZeroShotMovementAgent(prediction_seconds=5)
crowd_agent = CrowdDensitySurgeAgent(surge_threshold_per_sec=1.5)
anomaly_agent = BehavioralAnomalyAgent(anomaly_sensitivity_threshold=3.0)

db_conn = GraphDB(snapshot_path="data/snapshots/prod_graph.json")

missing_agent = MissingChildPredictiveAgent(
    movement_agent_instance=movement_agent,
    anomaly_agent_instance=anomaly_agent,
    graph_database_connection=db_conn,
)

crowd_safety_agent = CrowdSafetyPredictiveAgent(
    surge_velocity_threshold=1.5,
    structural_capacity=30,
)

civic_order_agent = CivicOrderPredictiveAgent(
    loitering_time_threshold_sec=10.0,
)

# ---------------------------------------------------------------------------
# In‑memory buffers – mirroring what `main.py` previously managed.
# ---------------------------------------------------------------------------
TRACKING_RAM_BUFFER: Dict[str, List[Any]] = {}
CAMERA_ACTIVE_PEOPLE: Dict[str, Dict[str, float]] = {}
CAMERA_COUNT_HISTORY: Dict[str, List[Any]] = {}
ENTITY_PACKET_COUNT: Dict[str, int] = {}

# ---------------------------------------------------------------------------
# Pydantic models for request validation
# ---------------------------------------------------------------------------
class TelemetryPacket(BaseModel):
    camera_id: str
    entity_id: str
    x_meters: float
    y_meters: float
    frame_timestamp: float

# ---------------------------------------------------------------------------
# Helper utilities – keep endpoint bodies concise.
# ---------------------------------------------------------------------------
def _update_buffers(packet: TelemetryPacket) -> int:
    """Update global buffers with a new telemetry packet.
    Returns the current count of active entities for the packet's camera.
    """
    eid = packet.entity_id
    cid = packet.camera_id
    ts = packet.frame_timestamp
    coords = (packet.x_meters, packet.y_meters)

    # Store raw trajectory (cap at 50 points per entity)
    TRACKING_RAM_BUFFER.setdefault(eid, []).append((coords[0], coords[1], ts))
    if len(TRACKING_RAM_BUFFER[eid]) > 50:
        TRACKING_RAM_BUFFER[eid].pop(0)

    # Update per‑camera active‑people map (2‑second sliding window)
    CAMERA_ACTIVE_PEOPLE.setdefault(cid, {})[eid] = ts
    cutoff = ts - 2.0
    CAMERA_ACTIVE_PEOPLE[cid] = {e: t for e, t in CAMERA_ACTIVE_PEOPLE[cid].items() if t > cutoff}
    current_count = len(CAMERA_ACTIVE_PEOPLE[cid])

    # Keep short history of occupancy counts for surge calculations
    CAMERA_COUNT_HISTORY.setdefault(cid, []).append((float(current_count), ts))
    if len(CAMERA_COUNT_HISTORY[cid]) > 50:
        CAMERA_COUNT_HISTORY[cid].pop(0)

    # Increment packet counter – triggers background analysis every 15 packets
    ENTITY_PACKET_COUNT[eid] = ENTITY_PACKET_COUNT.get(eid, 0) + 1

    return current_count

async def _maybe_trigger_background(entity_id: str, camera_id: str) -> None:
    """Launch background analysis for movement forecasting and anomaly detection.
    Called when the packet count for an entity reaches a multiple of 15.
    """
    if ENTITY_PACKET_COUNT.get(entity_id, 0) % 15 != 0:
        return

    async def _analysis() -> None:
        history = TRACKING_RAM_BUFFER.get(entity_id, [])
        pure_coords = [(pt[0], pt[1]) for pt in history]
        # Movement forecast (needs >=8 points)
        if len(pure_coords) >= 8:
            loop = asyncio.get_event_loop()
            move_res = await loop.run_in_executor(
                None, movement_agent.forecast_track, entity_id, pure_coords
            )
            if isinstance(move_res, dict) and move_res.get("status") == "SUCCESS":
                print(f"[BG] Movement forecast for {entity_id}: {move_res['predicted_trajectory'][0]}")
        # Anomaly detection (needs >=16 points)
        if len(history) >= 16:
            anomaly_res = anomaly_agent.evaluate_entity_tracklet(entity_id, history)
            print(
                f"[BG] Anomaly for {entity_id}: score={anomaly_res['anomaly_score']}, status={anomaly_res['status']}"
            )
            if anomaly_res.get("status") == "BEHAVIORAL_ANOMALY_DETECTED":
                print(f"[ALERT] Behavioral anomaly on {entity_id} – score {anomaly_res['anomaly_score']}")

    asyncio.create_task(_analysis())

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/telemetry")
async def ingest_telemetry(packet: TelemetryPacket) -> Dict[str, Any]:
    """Receive a telemetry packet, update buffers, and optionally trigger analysis.
    Returns a simple status payload.
    """
    try:
        current_count = _update_buffers(packet)
        await _maybe_trigger_background(packet.entity_id, packet.camera_id)
        # Quick surge check – useful for real‑time alerts.
        surge = crowd_agent.monitor_surge_rate(current_count, packet.frame_timestamp)
        if surge.get("status") == "CRITICAL_SURGE_ALERT":
            print(
                f"[ALERT] Critical crowd surge on {packet.camera_id}: {surge['surge_rate_per_sec']:.2f}/s"
            )
        return {"status": "QUEUED"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/v1/usecases/missing/{entity_id}")
async def missing_usecase(entity_id: str, q: str = "") -> Dict[str, Any]:
    """Execute the missing‑child predictive pipeline for a given entity.
    Returns whatever the `MissingChildPredictiveAgent` produces.
    """
    try:
        if entity_id not in TRACKING_RAM_BUFFER:
            return {
                "status": "GATHERING_CONTEXT",
                "message": f"No telemetry history for entity {entity_id}",
            }
        history = TRACKING_RAM_BUFFER[entity_id]
        return missing_agent.execute_investigative_pipeline(entity_id, history, user_query=q)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/v1/usecases/crowd/{camera_id}")
async def crowd_usecase(camera_id: str, q: str = "") -> Dict[str, Any]:
    """Run the crowd‑safety analysis for a camera.
    Generates a synthetic density map and delegates to the `CrowdSafetyPredictiveAgent`.
    """
    try:
        if camera_id not in CAMERA_ACTIVE_PEOPLE:
            return {
                "status": "INITIALIZING_BUFFER",
                "camera_id": camera_id,
                "message": "No active telemetry for this camera.",
            }
        current_count = len(CAMERA_ACTIVE_PEOPLE.get(camera_id, {}))
        history = CAMERA_COUNT_HISTORY.get(camera_id, [])
        import numpy as np
        density = np.random.rand(20, 20) * 0.1
        if current_count > 0:
            density += current_count * 0.05
        res = crowd_safety_agent.evaluate_choke_point_metrics(
            camera_id=camera_id,
            current_occupant_count=float(current_count),
            historical_counts_timeline=history,
            raw_csrnet_density_map=density,
            user_query=q,
        )
        if "metrics" in res:
            res["metrics"]["occupancy_density_count"] = float(
                res["metrics"]["occupancy_density_count"]
            )
            res["metrics"]["surge_velocity_rate"] = float(
                res["metrics"]["surge_velocity_rate"]
            )
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/v1/usecases/civic/traffic/{vehicle_id}")
async def traffic_compliance(vehicle_id: str, q: str = "") -> Dict[str, Any]:
    """Assess traffic‑compliance for a vehicle, including pedestrian helmet context.
    """
    try:
        pedestrians = ["WILDTRACK_PED_01", "WILDTRACK_PED_02", "WILDTRACK_PED_03"]
        helmets = {"WILDTRACK_PED_01": True, "WILDTRACK_PED_02": False, "WILDTRACK_PED_03": True}
        camera_id = "cam7"
        nd = db_conn._graph.nodes.get(vehicle_id, {})
        cams = list(nd.get("camera_ids", []))
        if cams:
            camera_id = cams[0]
        return civic_order_agent.process_traffic_compliance(
            vehicle_id, pedestrians, helmets, user_query=q, camera_id=camera_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/v1/usecases/civic/compliance/{entity_id}")
async def civic_compliance(entity_id: str, q: str = "") -> Dict[str, Any]:
    """Run the civic‑order compliance check for a generic entity (e.g., a person).
    Provides a mock abandoned‑object event if the telemetry history is short.
    """
    try:
        history = TRACKING_RAM_BUFFER.get(entity_id, [])
        if len(history) < 2:
            import time
            history = [
                (2.1, 3.4, time.time() - 12.0),
                (2.2, 3.5, time.time()),
            ]
        abandoned_event = [
            {"event_type": "OBJECT_ABANDONED", "class_label": "trash", "coordinates": (2.15, 3.45)}
        ]
        camera_id = "cam1"
        nd = db_conn._graph.nodes.get(entity_id, {})
        cams = list(nd.get("camera_ids", []))
        if cams:
            camera_id = cams[0]
        return civic_order_agent.process_civic_compliance(
            entity_id, history, abandoned_event, user_query=q, camera_id=camera_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ---------------------------------------------------------------------------
# Simple health check – useful for orchestration scripts.
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Run the ASGI server when executed directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
