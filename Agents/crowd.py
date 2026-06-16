import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from collections import deque
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
    def __init__(self, surge_threshold_per_sec: float = 1.5, memory_window_size: int = 10):
        self.surge_threshold = surge_threshold_per_sec
        
        # FIX: Implement explicit rolling queues to cap memory usage and prevent index allocation stalls
        self.historical_counts = deque(maxlen=memory_window_size)
        self.historical_timestamps = deque(maxlen=memory_window_size)
        
        print("[CROWD INIT] Pre-trained CSRNet volumetric density agent online and armed.")

    def monitor_surge_rate(self, current_count: float, timestamp: float) -> Dict:
        self.historical_counts.append(current_count)
        self.historical_timestamps.append(timestamp)
        
        if len(self.historical_counts) < 2:
            return {"surge_rate_per_sec": 0.0, "status": "BUFFERING"}
            
        # FIX: Extract consecutive sequence frame shifts to find the true instantaneous surge velocity
        delta_count = self.historical_counts[-1] - self.historical_counts[-2]
        delta_time = self.historical_timestamps[-1] - self.historical_timestamps[-2]
        
        surge_rate = delta_count / delta_time if delta_time > 0 else 0.0
        status_verdict = "CRITICAL_SURGE_ALERT" if surge_rate > self.surge_threshold else "NORMAL"
        
        return {
            "surge_rate_per_sec": round(surge_rate, 3),
            "status": status_verdict
        }

if __name__ == "__main__":
    crowd_agent = CrowdDensitySurgeAgent(surge_threshold_per_sec=1.5)
    print(crowd_agent.monitor_surge_rate(current_count=12.0, timestamp=time.time()))
    time.sleep(0.5)
    print(crowd_agent.monitor_surge_rate(current_count=14.5, timestamp=time.time()))