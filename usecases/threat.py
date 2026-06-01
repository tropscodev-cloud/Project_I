import os
import time
from typing import List, Dict, Tuple, Any

class ThreatNetworkPredictiveAgent:
    def __init__(self, behavioral_threat_threshold: float = 3.0):
        self.threat_threshold = behavioral_threat_threshold
        print("[STRATEGIC AGENT] Coordinated Threat Network Mapping Agent initialized.")
        print(f"[STRATEGIC AGENT] Behavioral screening rules armed: Threshold Score > {self.threat_threshold}")

    def execute_threat_network_discovery(self, 
                                         target_entity_id: str, 
                                         base_anomaly_agent_output: Dict[str, Any], 
                                         louvain_community_clusters: Dict[int, List[str]], 
                                         graph_proximity_network: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_processing_time = time.time()
        
        anomaly_score = base_anomaly_agent_output.get("anomaly_score", 0.0)
        anomaly_status = base_anomaly_agent_output.get("status", "NORMAL")

        target_cell_id = -1
        discovered_cell_members = []
        for cell_id, members in louvain_community_clusters.items():
            if target_entity_id in members:
                target_cell_id = cell_id
                discovered_cell_members = [m for m in members if m != target_entity_id]
                break

        high_risk_associates = []
        for associate in graph_proximity_network:
            if associate.get("confidence_score", 0.0) >= 0.70:
                high_risk_associates.append({
                    "entity_id": associate["entity_id"],
                    "relationship_confidence": associate["confidence_score"],
                    "total_contact_seconds": associate.get("total_duration_sec", 0.0),
                    "operational_role": "Identified Co-Presence Associate"
                })

        is_coordinated_threat = False
        verdict_tier = "GREEN_MONITORING"
        
        if anomaly_status == "BEHAVIORAL_ANOMALY_DETECTED" and (high_risk_associates or discovered_cell_members):
            is_coordinated_threat = True
            verdict_tier = "RED_COORDINATED_THREAT_ALERT"

        suspect_pattern_clause = "Normal, independent walking vectors registered across active profiles."
        if verdict_tier == "RED_COORDINATED_THREAT_ALERT":
            accomplice_list = discovered_cell_members if discovered_cell_members else [a["entity_id"] for a in high_risk_associates]
            suspect_pattern_clause = (
                f"COORDINATED THREAT MATRIX REVEALED: Target profile is exhibiting highly erratic, "
                f"suspicious pacing loops (Kinematic Score: {anomaly_score}). Cross-referencing our local "
                f"GraphDB exposes an active network connection to cell nodes: {accomplice_list}. "
                f"This suggests a coordinated scouting pattern or illicit team operation."
            )

        cognitive_briefing = (
            f"**URG-IS HIGH-STAKES SECURITY INTELLIGENCE REPORT**\n"
            f"**Flagged Suspect ID:** {target_entity_id}\n"
            f"**Kinematic Threat Profile:** {anomaly_status} (Score: {anomaly_score})\n\n"
            f"**Network Topology Cell Analysis:**\n"
            f"{suspect_pattern_clause}\n\n"
            f"**Sovereign Predictive Next Steps:**\n"
            f"**Operational Directive:** Flag accomplice nodes across all camera screens immediately. "
            f"Begin logging shared trajectory overlays to capture their next coordination point automatically."
        )

        execution_latency_ms = (time.time() - start_processing_time) * 1000

        return {
            "status": "SUCCESS",
            "agent_latency_ms": round(execution_latency_ms, 2),
            "target_id": target_entity_id,
            "threat_classification": verdict_tier,
            "network_context": {
                "louvain_cell_id": target_cell_id,
                "identified_cell_accomplices": discovered_cell_members,
                "direct_proximity_associates": high_risk_associates
            },
            "chatbot_payload": cognitive_briefing
        }

if __name__ == "__main__":
    threat_agent = ThreatNetworkPredictiveAgent(behavioral_threat_threshold=3.0)

    print("\n[STEP 1] Evaluating normal, baseline pedestrian behaviors...")
    mock_base_normal_output = {"anomaly_score": 1.5, "status": "NORMAL"}
    mock_empty_graph_network = []
    mock_base_louvain_partitions = {0: ["WILDTRACK_PED_42", "WILDTRACK_PED_05"], 1: ["WILDTRACK_PED_99"]}

    normal_payload = threat_agent.execute_threat_network_discovery(
        target_entity_id="WILDTRACK_PED_42",
        base_anomaly_agent_output=mock_base_normal_output,
        louvain_community_clusters=mock_base_louvain_partitions,
        graph_proximity_network=mock_empty_graph_network
    )
    print(normal_payload["chatbot_payload"])
    print("="*75)

    print("\n[STEP 2] Simulating high-stakes security threat (Kinematic Anomaly + Graph Network Match)...")
    mock_base_critical_output = {"anomaly_score": 415.17, "status": "BEHAVIORAL_ANOMALY_DETECTED"}
    
    mock_active_graph_relations = [
        {"entity_id": "WILDTRACK_PED_99", "confidence_score": 0.85, "total_duration_sec": 124.0}
    ]
    mock_louvain_partitions = {
        0: ["WILDTRACK_PED_42", "WILDTRACK_PED_99"],
        1: ["WILDTRACK_PED_01", "WILDTRACK_PED_02"]
    }

    threat_payload = threat_agent.execute_threat_network_discovery(
        target_entity_id="WILDTRACK_PED_42",
        base_anomaly_agent_output=mock_base_critical_output,
        louvain_community_clusters=mock_louvain_partitions,
        graph_proximity_network=mock_active_graph_relations
    )
    
    print("\n" + "="*75)
    print("           DEPLOYED THREAT MODULE AGENT: REAL-TIME RESPONSE MATRIX       ")
    print("="*75)
    print(threat_payload["chatbot_payload"])
    print("="*75)
    print(f"Agent Analytics Computational Latency: {threat_payload['agent_latency_ms']} ms")
    print("="*75)