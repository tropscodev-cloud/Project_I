import sys
import time
import numpy as np
sys.path.append('.')

import pipeline_state as PS
from usecases.crowd import CrowdSafetyPredictiveAgent

print("Running fallback simulation...")
local_crowd_agent = CrowdSafetyPredictiveAgent(surge_velocity_threshold=1.5, structural_capacity=30)
curr_count = 14.0
mock_hist = [(max(0.0, curr_count - 1.0), time.time() - 1.0), (curr_count, time.time())]
sim_map = np.random.rand(20, 20) * 0.1
sim_map = sim_map + (curr_count * 0.05)

t0 = time.time()
try:
    payload = local_crowd_agent.evaluate_choke_point_metrics("cam1", curr_count, mock_hist, sim_map)
    print(f"Success! Time taken: {time.time() - t0:.4f}s")
    print("Payload keys:", list(payload.keys()))
    print("Chatbot payload:")
    print(payload["chatbot_payload"])
except Exception as e:
    import traceback
    print("FAILED with exception:")
    traceback.print_exc()
