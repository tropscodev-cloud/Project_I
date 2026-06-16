import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agent import generate_dynamic_usecase_briefing

ctx = {
    "camera_id": "cam1",
    "occupancy": 12.0,
    "surge_rate": 0.5,
    "compression": 25.0,
    "verdict": "NORMAL",
    "directives": []
}

print("Calling generate_dynamic_usecase_briefing...")
res = generate_dynamic_usecase_briefing("crowd", ctx, user_query="Check crowd density for cam1")
print("Result:")
print(res)
