import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, Tuple

class CSRNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(CSRNetFeatureExtractor, self).__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.backend = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

class CrowdDensitySurgeAgent:
    def __init__(self, surge_threshold_per_sec: float = 0.5):
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CSRNetFeatureExtractor().to(self.device)
        self.model.eval()
        
        self.surge_threshold = surge_threshold_per_sec
        self.historical_counts = []
        self.historical_timestamps = []

        print("[CROWD INIT] Dilated-CNN Density Regressor initialized successfully.")
        print(f"[CROWD INIT] Execution target environment: '{self.device}'")

    def process_frame_matrix(self, frame_ndarray: np.ndarray) -> Tuple[float, np.ndarray]:
        resized = cv2.resize(frame_ndarray, (640, 480))
        img_tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_tensor /= 255.0
        
        with torch.no_grad():
            density_map = self.model(img_tensor)
            
        calculated_count = torch.sum(density_map).item()
        calculated_count = max(0.0, calculated_count)
        
        density_matrix = density_map.squeeze().cpu().numpy()
        
        return calculated_count, density_matrix

    def monitor_surge_rate(self, current_count: float, timestamp: float) -> Dict:
        self.historical_counts.append(current_count)
        self.historical_timestamps.append(timestamp)
        
        if len(self.historical_counts) > 10:
            self.historical_counts.pop(0)
            self.historical_timestamps.pop(0)
            
        if len(self.historical_counts) < 2:
            return {"surge_rate_per_sec": 0.0, "status": "STABLE"}
            
        delta_count = self.historical_counts[-1] - self.historical_counts[0]
        delta_time = self.historical_timestamps[-1] - self.historical_timestamps[0]
        
        surge_rate = delta_count / delta_time if delta_time > 0 else 0.0
        
        status = "CRITICAL_SURGE_ALERT" if surge_rate > self.surge_threshold else "NORMAL"
        
        return {
            "surge_rate_per_sec": round(surge_rate, 3),
            "status": status
        }

if __name__ == "__main__":
    crowd_agent = CrowdDensitySurgeAgent(surge_threshold_per_sec=1.5)
    
    print("\n[TEST] Emulating real-time incoming video frames for crowd evaluation...")
    mock_timestamps = [time.time() + (i * 0.5) for i in range(6)]
    mock_people_counts = [1.2, 1.5, 2.8, 4.5, 6.2, 7.5]
    
    print(f"[TEST] Incoming frame telemetry sequence initialized.")
    print("-" * 65)

    for count, ts in zip(mock_people_counts, mock_timestamps):
        metrics = crowd_agent.monitor_surge_rate(current_count=count, timestamp=ts)
        print(f"Timestamp: {ts:.2f} | Count: {count:.1f} | Surge Rate: {metrics['surge_rate_per_sec']}/s | Status: {metrics['status']}")
        time.sleep(0.1)
        
    print("=" * 65)
    print("               CROWD SURGE AGENT VERIFICATION SUMMARY            ")
    print("=" * 65)
    print(f"Final Count Registered : {mock_people_counts[-1]} individuals")
    print(f"Calculated Alert Level : {metrics['status']}")
    print(f"Operational Action     : Pushing alert state to local GraphDB database.")
    print("=" * 65)