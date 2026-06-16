import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Agents.movement import ZeroShotMovementAgent
from Agents.anomaly import BehavioralAnomalyAgent
from core.graph.graph_db import GraphDB
from agent import generate_dynamic_usecase_briefing


class MissingChildPredictiveAgent:
    def __init__(self, movement_agent_instance: Any, anomaly_agent_instance: Any, graph_database_connection: Any = None):
        self.movement_agent = movement_agent_instance
        self.anomaly_agent = anomaly_agent_instance
        self.db_conn = graph_database_connection
        print("[STRATEGIC AGENT] Missing Child Recovery Use-Case Agent fully initialized.")
        print("[STRATEGIC AGENT] Predictive Intelligence Layer actively backed by Foundation Models.")

    def execute_investigative_pipeline(
        self,
        target_child_id: str,
        raw_coordinate_stream: List[Tuple[float, float, float]],
        user_query: str = ""
    ) -> Dict[str, Any]:
        start_time = time.time()

        pure_coords_only = [(pt[0], pt[1]) for pt in raw_coordinate_stream]

        if len(pure_coords_only) < 8:
            return {
                "status": "GATHERING_CONTEXT",
                "message": f"Historical data queue building. Current depth: {len(pure_coords_only)}/8 tokens."
            }

        movement_forecast = self.movement_agent.forecast_track(
            entity_id=target_child_id,
            sanitized_history=pure_coords_only
        )

        behavior_profile = self.anomaly_agent.evaluate_entity_tracklet(
            entity_id=target_child_id,
            raw_coordinate_stream=raw_coordinate_stream
        )

        isolated_suspect_leads = []
        if self.db_conn:
            graph_data = self.db_conn.get_person_graph(target_child_id)
            if graph_data:
                for conn in graph_data.get("connections", []):
                    if conn["confidence"] >= 0.70:
                        isolated_suspect_leads.append({
                            "entity_id": conn["person_id"],
                            "relationship_class": conn["relationship"].replace("_", " ").title(),
                            "confidence_score": conn["confidence"],
                            "threat_status": "SUSPECT_IDENTIFIED_VIA_RELATIONAL_PROXIMITY"
                        })

        predicted_path = movement_forecast.get("predicted_trajectory", [])
        last_known_position = pure_coords_only[-1]

        camera_id = "cam1"
        if self.db_conn:
            try:
                nd = self.db_conn._graph.nodes[target_child_id]
                cams = list(nd.get("camera_ids", []))
                if cams:
                    camera_id = cams[0]
            except Exception:
                pass

        from config.settings import get_camera_location_name
        setting_name = get_camera_location_name(camera_id)

        ctx = {
            "target_id": target_child_id,
            "camera_id": camera_id,
            "setting_name": setting_name,
            "last_seen": last_known_position,
            "projected_path_trail": predicted_path,
            "kinematic_anomaly_score": behavior_profile.get("anomaly_score", 0.0),
            "isolated_suspects": isolated_suspect_leads,
        }
        chatbot_briefing_summary = generate_dynamic_usecase_briefing("missing", ctx, user_query=user_query)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "status": "SUCCESS",
            "latency_ms": round(latency_ms, 2),
            "target_id": target_child_id,
            "predictions": {
                "last_seen": last_known_position,
                "projected_path_trail": predicted_path,
                "kinematic_anomaly_score": behavior_profile.get("anomaly_score", 0.0)
            },
            "investigative_leads": {
                "isolated_suspects": isolated_suspect_leads,
                "optimal_interception_point": predicted_path[-1] if predicted_path else last_known_position
            },
            "chatbot_payload": chatbot_briefing_summary
        }


if __name__ == "__main__":
    base_movement_agent = ZeroShotMovementAgent(prediction_seconds=5)
    base_anomaly_agent = BehavioralAnomalyAgent(anomaly_sensitivity_threshold=3.0)
    db = GraphDB(snapshot_path="data/snapshots/prod_graph.json")

    usecase_agent = MissingChildPredictiveAgent(
        movement_agent_instance=base_movement_agent,
        anomaly_agent_instance=base_anomaly_agent,
        graph_database_connection=db
    )

    # Timestamps spaced 0.5s apart — realistic walking-speed kinematics
    base_ts = time.time() - 8.0
    mock_real_telemetry_history = [
        (0.5, 1.2, base_ts + 0.0),
        (0.7, 1.5, base_ts + 0.5),
        (0.9, 1.8, base_ts + 1.0),
        (1.1, 2.1, base_ts + 1.5),
        (1.3, 2.4, base_ts + 2.0),
        (1.5, 2.7, base_ts + 2.5),
        (1.7, 3.0, base_ts + 3.0),
        (1.9, 3.3, base_ts + 3.5),
        (2.1, 3.6, base_ts + 4.0),
        (2.3, 3.9, base_ts + 4.5),
        (2.5, 4.2, base_ts + 5.0),
        (2.7, 4.5, base_ts + 5.5),
        (2.9, 4.8, base_ts + 6.0),
        (3.2, 5.1, base_ts + 6.5),
        (3.7, 5.4, base_ts + 7.0),
        (4.2, 5.7, base_ts + 7.5),
    ]

    print("\n[EXEC] Running the continuous Use-Case Pipeline on target data tracks...")
    final_output = usecase_agent.execute_investigative_pipeline(
        target_child_id="19",
        raw_coordinate_stream=mock_real_telemetry_history
    )

    print("\n" + "=" * 75)
    print("        PREDICTIVE INTELLIGENCE AGENT: INVESTIGATIVE TEXT RESPONSE       ")
    print("=" * 75)
    print(final_output["chatbot_payload"])
    print("=" * 75)
    print(f"Agent Analytics Computational Latency: {final_output['latency_ms']} ms")
    print("=" * 75)