import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from Agents.movement import ZeroShotMovementAgent
from Agents.anomaly import BehavioralAnomalyAgent
from core.graph.graph_db import GraphDB

class MissingChildPredictiveAgent:
    def __init__(self, movement_agent_instance: Any, anomaly_agent_instance: Any, graph_database_connection: Any = None):
        self.movement_agent = movement_agent_instance
        self.anomaly_agent = anomaly_agent_instance
        self.db_conn = graph_database_connection
        print("[STRATEGIC AGENT] Missing Child Recovery Use-Case Agent fully initialized.")
        print("[STRATEGIC AGENT] Predictive Intelligence Layer actively backed by Foundation Models.")

    def execute_investigative_pipeline(self, target_child_id: str, raw_coordinate_stream: List[Tuple[float, float, float]]) -> Dict[str, Any]:
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
            raw_history=raw_coordinate_stream
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
        
        suspect_brief = "No immediate high-risk interpersonal anomalies registered in the sub-graph."
        if isolated_suspect_leads:
            suspect_brief = f"CRITICAL TARGET IDENTIFIED: Node `{isolated_suspect_leads[0]['entity_id']}` is flagged as a primary suspect based on proximity history."

        chatbot_briefing_summary = (
            f"**URG-IS Predictive Use-Case Briefing: Missing Child Search**\n\n"
            f"**Target Identity ID:** `{target_child_id}`\n"
            f"**Last Known Coordinates:** X={last_known_position[0]}m, Y={last_known_position[1]}m\n\n"
            f"**1. Graph Database Suspect Vector:**\n"
            f"{suspect_brief}\n\n"
            f"**2. Foundation-Backed Path Forecast (Amazon Chronos):**\n"
            f"The underlying transformer model has analyzed trailing velocity arrays. "
            f"**Target Interception Zone:** The subject is projected to cross physical coordinate point **X={predicted_path[-1][0]}m, Y={predicted_path[-1][1]}m** within 5 seconds. "
            f"Deploy intercept operators directly to this location for field recovery."
        )

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

    mock_real_telemetry_history = [
        (0.5, 1.2, 1779959333.54), (0.7, 1.5, 1779959333.57),
        (0.9, 1.8, 1779959333.61), (1.1, 2.1, 1779959333.64),
        (1.3, 2.4, 1779959333.67), (1.5, 2.7, 1779959333.71),
        (1.7, 3.0, 1779959333.74), (1.9, 3.3, 1779959333.77),
        (2.1, 3.6, 1779959333.80), (2.3, 3.9, 1779959333.84),
        (2.5, 4.2, 1779959333.87), (2.7, 4.5, 1779959333.90),
        (2.9, 4.8, 1779959334.00), (3.2, 5.1, 1779959334.04),
        (3.7, 5.4, 1779959334.08), (4.2, 5.7, 1779959334.12)
    ]

    print("\n[EXEC] Running the continuous Use-Case Pipeline on target data tracks...")
    final_output = usecase_agent.execute_investigative_pipeline(
        target_child_id="19", 
        raw_coordinate_stream=mock_real_telemetry_history
    )

    print("\n" + "="*75)
    print("        PREDICTIVE INTELLIGENCE AGENT: INVESTIGATIVE TEXT RESPONSE       ")
    print("="*75)
    print(final_output["chatbot_payload"])
    print("="*75)
    print(f"Agent Analytics Computational Latency: {final_output['latency_ms']} ms")
    print("="*75)