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
        # Crucial extraction components
        x_pts = window_data[:, 0]
        y_pts = window_data[:, 1]
        ts_pts = window_data[:, 2]

        # Explicitly derive positional shifts across frames
        dt = np.diff(ts_pts)
        dt[dt <= 0.001] = 0.033  # Safe video frame rate interval fallback (approx 30fps)

        dx = np.diff(x_pts)
        dy = np.diff(y_pts)
        distances = np.sqrt(dx**2 + dy**2)
        velocities = distances / dt

        accelerations = np.diff(velocities) / dt[:-1] if len(velocities) > 1 else np.zeros(1)
        acceleration_variance = np.var(accelerations) if len(accelerations) > 0 else 0.0

        total_path_length = np.sum(distances)
        net_displacement = np.sqrt((x_pts[-1] - x_pts[0])**2 + (y_pts[-1] - y_pts[0])**2)
        
        # Guard against stagnant or low-movement noise signals blowing up variance indices
        if net_displacement < 0.1 or total_path_length < 0.15:
            return 0.01

        # Complexity metric: Path length vs straight-line displacement
        efficiency_ratio = total_path_length / max(0.01, net_displacement)
        raw_anomaly_index = acceleration_variance * efficiency_ratio

        return float(raw_anomaly_index)

    def evaluate_entity_tracklet(self, entity_id: str, raw_coordinate_stream: List[Tuple[float, float, float]]) -> Dict:
        if len(raw_coordinate_stream) < self.window_size:
            return {
                "entity_id": entity_id,
                "status": "INITIALIZING_STREAM",
                "anomaly_score": 0.0,
                "message": f"Buffering trajectory points. Current depth: {len(raw_coordinate_stream)}/{self.window_size}"
            }

        # FIX: Force float64 allocation explicitly to capture Unix epoch sub-second timestamp decimals
        matrix = np.array(raw_coordinate_stream, dtype=np.float64)
        active_window = matrix[-self.window_size:]

        anomaly_score = self.compute_kinematic_variance(active_window)
        status_verdict = "BEHAVIORAL_ANOMALY_DETECTED" if anomaly_score > self.sensitivity else "NORMAL"

        return {
            "entity_id": entity_id,
            "status": status_verdict,
            "anomaly_score": round(anomaly_score, 3),
            "evaluated_window_size": self.window_size,
            "timestamp": time.time()
        }

if __name__ == "__main__":
    anomaly_agent = BehavioralAnomalyAgent(anomaly_sensitivity_threshold=3.0)
    t_base = time.time()
    normal_tracklet = [(1.0 + (i * 0.1), 2.0 + (i * 0.1), t_base + (i * 1.0)) for i in range(16)]
    res = anomaly_agent.evaluate_entity_tracklet("WILDTRACK_PED_12", normal_tracklet)
    print(f"Standalone Test Executed: {res}")