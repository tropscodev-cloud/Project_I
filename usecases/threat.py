import os
import sys
import time
from typing import List, Dict, Tuple, Any

class ThreatNetworkPredictiveAgent:
    def __init__(self, behavioral_threat_threshold: float = 3.0):
        self.threat_threshold = behavioral_threat_threshold

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
            if associate.get("confidence_score", 0.0) >= 0.20:
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

        if verdict_tier == "RED_COORDINATED_THREAT_ALERT":
            narrative = (
                "Warning: Target profile exhibits suspicious coordinated walking pacing. "
                "Cross-referencing GraphDB records reveals active linkage to multiple cell nodes. "
                "This indicates potential team scouting or coordinated patrol pattern."
            )
        else:
            narrative = (
                "The target profile is currently exhibiting regular movement patterns. "
                "No anomalous social structures or high-risk co-presence linkages have been triggered."
            )

        # Build clean mathematical proof without emojis or asterisks
        associates_proof = []
        for assoc in high_risk_associates:
            e_id = assoc["entity_id"]
            conf = assoc["relationship_confidence"]
            dur = assoc["total_contact_seconds"]
            # Confidence formulation: weight = duration / 60 seconds (min duration ratio normalized)
            associates_proof.append(
                f"Link to P{e_id}: confidence weight is {round(conf, 2)} based on {round(dur, 1)} seconds contact time"
            )
        
        associates_str = "; ".join(associates_proof) if associates_proof else "None identified"

        math_proof = (
            "Mathematical proof of hidden associations:\n"
            f"1. Target Person ID: P{target_entity_id}\n"
            f"2. Kinematic Anomaly Score: {round(anomaly_score, 2)} (Threshold: {self.threat_threshold})\n"
            f"3. Louvain community partition cluster ID: {target_cell_id} with {len(discovered_cell_members)} other members\n"
            f"4. Co-presence links probability list: {associates_str}"
        )

        cognitive_briefing = (
            "Hidden Associates Threat Discovery Report\n\n"
            f"Verdict: {verdict_tier}\n\n"
            f"Summary: {narrative}\n\n"
            f"{math_proof}\n\n"
            "Directive: Flag identified contact nodes on dashboard and record trajectory overlays."
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