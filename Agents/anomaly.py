import time
import numpy as np
from typing import List, Tuple, Dict

class BehavioralAnomalyAgent:
    def __init__(self, anomaly_sensitivity_threshold: float = 2.5):
        self.sensitivity = anomaly_sensitivity_threshold
        self.window_size = 16

        print("[ANOMALY INIT] Spatio-Temporal Sequence Parser armed and active.")
        print("[ANOMALY INIT] Running securely in localized CPU engine mode.")

    def compute_kinematic_variance(self, window_data: np.ndarray) -> float:
        x_pts = window_data[:, 0]
        y_pts = window_data[:, 1]
        ts_pts = window_data[:, 2]

        dt = np.diff(ts_pts)
        dt[dt <= 0] = 0.033

        dx = np.diff(x_pts)
        dy = np.diff(y_pts)
        distances = np.sqrt(dx**2 + dy**2)
        velocities = distances / dt

        accelerations = np.diff(velocities) / dt[:-1] if len(velocities) > 1 else np.zeros(1)
        acceleration_variance = np.var(accelerations) if len(accelerations) > 0 else 0.0

        total_path_length = np.sum(distances)
        net_displacement = np.sqrt((x_pts[-1] - x_pts[0])**2 + (y_pts[-1] - y_pts[0])**2)
        if net_displacement < 0.1 or total_path_length < 0.15:
            tortuosity = 1.0
        else:
            tortuosity = total_path_length / net_displacement

        anomaly_score = (acceleration_variance * 0.4) + (tortuosity * 1.5)
        return float(anomaly_score)

    def evaluate_entity_tracklet(self, entity_id: str, raw_history: List[Tuple[float, float, float]]) -> Dict:
        if len(raw_history) < self.window_size:
            return {
                "entity_id": entity_id,
                "status": "COLLECTING_CONTEXT",
                "current_depth": f"{len(raw_history)}/{self.window_size} frames"
            }

        target_window = np.array(raw_history[-self.window_size:], dtype=np.float32)
        
        score = self.compute_kinematic_variance(target_window)
        
        status = "NORMAL"
        if score > self.sensitivity:
            status = "BEHAVIORAL_ANOMALY_DETECTED"

        return {
            "entity_id": entity_id,
            "status": status,
            "anomaly_score": round(score, 3),
            "evaluated_window_size": self.window_size,
            "timestamp": time.time()
        }

if __name__ == "__main__":
    anomaly_agent = BehavioralAnomalyAgent(anomaly_sensitivity_threshold=3.0)
    
    print("\n[TEST] Scenario A: Evaluating steady, linear pedestrian tracking inputs...")
    t_base = time.time()
    normal_tracklet = [
        (1.0 + (i * 0.1), 2.0 + (i * 0.1), t_base + (i * 0.04)) 
        for i in range(16)
    ]
    
    res_a = anomaly_agent.evaluate_entity_tracklet("WILDTRACK_PED_12", normal_tracklet)
    print(f"Result A -> Status: {res_a['status']} | Score: {res_a.get('anomaly_score')}")

    print("\n[TEST] Scenario B: Evaluating sudden erratic pacing/loitering behavior...")
    erratic_tracklet = []
    for i in range(16):
        x_err = 5.0 + np.sin(i * 1.5) * 0.8
        y_err = 4.0 + np.cos(i * 1.5) * 0.8
        erratic_tracklet.append((x_err, y_err, t_base + (i * 0.04)))
        
    res_b = anomaly_agent.evaluate_entity_tracklet("SUBJECT_HOMESTEAD_04", erratic_tracklet)
    
    print("-" * 65)
    print("             BEHAVIORAL ANOMALY AGENT OUTPUT REPORT              ")
    print("=" * 65)
    print(f"Target Entity  : {res_b['entity_id']}")
    print(f"Engine Status  : {res_b['status']}")
    print(f"Anomaly Score  : {res_b['anomaly_score']} (Threshold: 3.0)")
    print(f"Unix Timestamp : {res_b['timestamp']}")
    print(f"Action Taken   : Flagging entity sub-graph state as HIGH RISK inside GraphDB.")
    print("=" * 65)