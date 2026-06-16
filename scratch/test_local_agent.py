import sys
import time
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import like usecases_page.py does
from usecases.crowd import CrowdSafetyPredictiveAgent
import pipeline_state as PS

local_crowd_agent = CrowdSafetyPredictiveAgent(surge_velocity_threshold=1.5, structural_capacity=30)
curr_count = float(PS.G.get("person_count", 14.0))
mock_hist = [(curr_count - 1.0, time.time() - 2.0), (curr_count, time.time())]

print("Calling evaluate_choke_point_metrics...")
try:
    payload = local_crowd_agent.evaluate_choke_point_metrics("cam1", curr_count, mock_hist, np.random.rand(20,20)*0.1, user_query="Check crowd density for cam1")
    print("Success! Payload chatbot_payload:")
    print(payload["chatbot_payload"])
except Exception as e:
    import traceback
    print("Failed with exception:")
    traceback.print_exc()
