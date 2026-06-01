import os
import time
from typing import List, Dict, Tuple, Any

class CivicOrderPredictiveAgent:
    def __init__(self, loitering_time_threshold_sec: float = 10.0):
        self.loitering_threshold = loitering_time_threshold_sec
        print("[STRATEGIC AGENT] Civic Order & Traffic Enforcement Agent fully initialized.")
        print(f"[STRATEGIC AGENT] Behavioral anomaly triggers armed: Loitering Threshold = {self.loitering_threshold}s")

    def process_traffic_compliance(self, 
                                   vehicle_id: str, 
                                   associated_pedestrian_ids: List[str], 
                                   helmet_detected_flags: Dict[str, bool]) -> Dict[str, Any]:
        start_time = time.time()
        violations_detected = []
        
        passenger_count = len(associated_pedestrian_ids)
        if passenger_count > 2:
            violations_detected.append({
                "type": "TRIPLE_RIDING_VIOLATION",
                "severity": "HIGH",
                "details": f"Detected {passenger_count} human entities clustered on single vehicle node {vehicle_id}."
            })

        for pid in associated_pedestrian_ids:
            has_helmet = helmet_detected_flags.get(pid, True)
            if not has_helmet:
                violations_detected.append({
                    "type": "HELMET_GAP_VIOLATION",
                    "severity": "CRITICAL",
                    "details": f"Entity `{pid}` registered missing helmet signature during active vehicle transit."
                })

        status_verdict = "TRAFFIC_INFRACTION_DETECTED" if violations_detected else "COMPLIANT"
        latency_ms = (time.time() - start_time) * 1000

        return {
            "status": "SUCCESS",
            "latency_ms": round(latency_ms, 3),
            "vehicle_identifier": vehicle_id,
            "compliance_status": status_verdict,
            "active_violations": violations_detected
        }

    def process_civic_compliance(self, 
                                 entity_id: str, 
                                 spatial_history_meters: List[Tuple[float, float, float]], 
                                 object_permanence_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_time = time.time()
        civic_violations = []

        for event in object_permanence_events:
            if event.get("event_type") == "OBJECT_ABANDONED" and event.get("class_label") == "trash":
                civic_violations.append({
                    "type": "LITTERING_VIOLATION",
                    "severity": "MEDIUM",
                    "coordinates": event.get("coordinates", (0.0, 0.0)),
                    "details": f"Entity `{entity_id}` broke object proximity boundaries, abandoning a trash signature container."
                })

        if len(spatial_history_meters) >= 2:
            elapsed_duration = spatial_history_meters[-1][2] - spatial_history_meters[0][2]
            if elapsed_duration >= self.loitering_threshold:
                xs = [pt[0] for pt in spatial_history_meters]
                ys = [pt[1] for pt in spatial_history_meters]
                net_displacement = ((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)**0.5
                
                if net_displacement < 1.5:
                    civic_violations.append({
                        "type": "LOITERING_NUISANCE_VIOLATION",
                        "severity": "LOW",
                        "details": f"Entity maintained spatial location loop for {round(elapsed_duration, 1)}s with negligible net displacement."
                    })

        status_verdict = "CIVIC_INFRACTION_DETECTED" if civic_violations else "COMPLIANT"
        latency_ms = (time.time() - start_time) * 1000

        infraction_summary = "Urban environment parameters within normal baseline. Civic compliance sustained."
        if civic_violations:
            infraction_summary = f"CIVIC DISORDER ISOLATED: Entity `{entity_id}` has been flagged for active city compliance infractions: {[v['type'] for v in civic_violations]}."

        cognitive_briefing = (
            f"**URG-IS SMART CITY CIVIC OPERATIONS REPORT**\n"
            f"**Monitored Citizen Handle:** {entity_id}\n"
            f"**Compliance Status:** {status_verdict}\n\n"
            f"**Behavioral Matrix Evaluation:**\n"
            f"{infraction_summary}\n\n"
            f"**Sovereign Predictive Enforcement Action:**\n"
            f"**Action Taken:** Violations logged directly to the local relationship graph. "
            f"System has linked this infraction to the citizen's behavioral profile token for dynamic penalty processing."
        )

        return {
            "status": "SUCCESS",
            "latency_ms": round(latency_ms, 3),
            "entity_id": entity_id,
            "compliance_status": status_verdict,
            "active_violations": civic_violations,
            "chatbot_payload": cognitive_briefing
        }

if __name__ == "__main__":
    agent = CivicOrderPredictiveAgent()

    print("\n[TEST 1] Running Multi-Passenger Vehicle Compliance Assessment...")
    vehicle_output = agent.process_traffic_compliance(
        vehicle_id="MOTORCYCLE_NODE_77",
        associated_pedestrian_ids=["WILDTRACK_PED_01", "WILDTRACK_PED_02", "WILDTRACK_PED_03"],
        helmet_detected_flags={"WILDTRACK_PED_01": True, "WILDTRACK_PED_02": False, "WILDTRACK_PED_03": True}
    )
    import json
    print(json.dumps(vehicle_output, indent=2))
    print("="*75)

    print("\n[TEST 2] Running Spatial Civic Nuisance Assessment...")
    mock_active_loiter_history = [
        (2.1, 3.4, time.time() - 12.0),
        (2.2, 3.5, time.time())
    ]
    mock_abandoned_object_event = [
        {"event_type": "OBJECT_ABANDONED", "class_label": "trash", "coordinates": (2.15, 3.45)}
    ]

    civic_output = agent.process_civic_compliance(
        entity_id="WILDTRACK_PED_42",
        spatial_history_meters=mock_active_loiter_history,
        object_permanence_events=mock_abandoned_object_event
    )
    print(civic_output["chatbot_payload"])
    print("="*75)