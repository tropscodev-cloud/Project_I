import os
import sys
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agent import generate_dynamic_usecase_briefing


class CrowdSafetyPredictiveAgent:
    def __init__(self, surge_velocity_threshold: float = 1.5, structural_capacity: int = 30):
        self.surge_threshold = surge_velocity_threshold
        self.max_capacity = structural_capacity
        print("[STRATEGIC AGENT] Crowd Safety & Stampede Prevention Agent active.")
        print(f"[STRATEGIC AGENT] Safety parameters armed: Threshold={self.surge_threshold}/s, Max Capacity={self.max_capacity}")

    def evaluate_choke_point_metrics(self,
                                     camera_id: str,
                                     current_occupant_count: float,
                                     historical_counts_timeline: List[Tuple[float, float]],
                                     raw_csrnet_density_map: np.ndarray,
                                     user_query: str = "") -> Dict[str, Any]:
        start_processing_time = time.time()

        if len(historical_counts_timeline) < 2:
            return {
                "status": "INITIALIZING_BUFFER",
                "camera_id": camera_id,
                "message": "Accumulating sequence context logs."
            }

        # Use last two entries in the timeline to compute a stable dt
        penultimate_count, penultimate_timestamp = historical_counts_timeline[-2]
        last_count, last_timestamp = historical_counts_timeline[-1]
        dt = float(last_timestamp - penultimate_timestamp)
        if dt <= 0.001:
            dt = 1.0  # Safe fallback to 1-second interval normalization

        delta_count = float(current_occupant_count - last_count)
        calculated_surge_rate = delta_count / dt

        total_hotspot_pixels = int(np.sum(raw_csrnet_density_map > 0.6))
        compression_risk_percentage = min(100.0, (current_occupant_count / self.max_capacity) * 100)

        status_verdict = "NORMAL"
        action_directives = []

        if current_occupant_count >= self.max_capacity:
            status_verdict = "CRITICAL_CAPACITY_BREACH"
            action_directives.extend(["EMERGENCY_GATE_RELEASE_OVERRIDE", "DISPATCH_RESPONSE_CELL_PHASE_1"])
        elif calculated_surge_rate >= self.surge_threshold:
            status_verdict = "CRITICAL_SURGE_ALERT"
            action_directives.extend(["TRIGGER_FLOW_DIVERSION_ALGORITHMS", "BROADCAST_AUDIO_WARNING_LOOP"])

        ctx = {
            "camera_id": camera_id,
            "occupancy": current_occupant_count,
            "surge_rate": calculated_surge_rate,
            "compression": compression_risk_percentage,
            "verdict": status_verdict,
            "directives": action_directives,
        }
        cognitive_briefing = generate_dynamic_usecase_briefing("crowd", ctx, user_query=user_query)

        execution_latency_ms = (time.time() - start_processing_time) * 1000

        return {
            "status": "SUCCESS",
            "agent_latency_ms": round(execution_latency_ms, 2),
            "camera_id": camera_id,
            "safety_verdict": status_verdict,
            "metrics": {
                "occupancy_density_count": round(current_occupant_count, 2),
                "surge_velocity_rate": round(calculated_surge_rate, 3),
                "hotspot_pixel_saturation": total_hotspot_pixels
            },
            "hardware_overrides": action_directives,
            "chatbot_payload": cognitive_briefing
        }


if __name__ == "__main__":
    safety_agent = CrowdSafetyPredictiveAgent(surge_velocity_threshold=5.0, structural_capacity=50)

    print("\n[STEP 1] Evaluating normal operational baseline crowd metrics...")
    t1 = time.time()
    mock_timeline_buffer = [(12.0, t1 - 2.0), (12.0, t1 - 1.0)]
    mock_csrnet_array = np.random.rand(20, 20) * 0.3

    normal_output = safety_agent.evaluate_choke_point_metrics(
        camera_id="cam1",
        current_occupant_count=12.5,
        historical_counts_timeline=mock_timeline_buffer,
        raw_csrnet_density_map=mock_csrnet_array
    )
    print(normal_output["chatbot_payload"])
    print("=" * 75)

    print("\n[STEP 2] Simulating high-speed flash crowd compression surge event...")
    t2 = time.time()
    mock_surge_timeline_buffer = [(12.0, t2 - 2.0), (12.0, t2 - 1.0)]
    mock_critical_csrnet_array = np.random.rand(20, 20) * 0.9

    surge_output = safety_agent.evaluate_choke_point_metrics(
        camera_id="cam1",
        current_occupant_count=24.0,
        historical_counts_timeline=mock_surge_timeline_buffer,
        raw_csrnet_density_map=mock_critical_csrnet_array
    )

    print("\n" + "=" * 75)
    print("           DEPLOYED CROWD SAFETY AGENT: REAL-TIME RESPONSE MATRIX        ")
    print("=" * 75)
    print(surge_output["chatbot_payload"])
    print("=" * 75)
    print(f"Agent Analytics Computational Latency: {surge_output['agent_latency_ms']} ms")
    print("=" * 75)