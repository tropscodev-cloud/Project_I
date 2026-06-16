import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agent import generate_dynamic_usecase_briefing

usecases = [
    ("threat", {
        "target_id": "42",
        "camera_id": "cam1",
        "setting_name": "Entrance & Foyer",
        "anomaly_score": 1.5,
        "anomaly_status": "NORMAL",
        "cell_id": -1,
        "cell_members": [],
        "associates": [],
    }, "Is this person suspicious?"),
    ("missing", {
        "target_id": "19",
        "camera_id": "cam1",
        "setting_name": "Entrance & Foyer",
        "last_seen": (2.3, 4.5),
        "projected_path_trail": [(2.5, 4.7), (2.7, 4.9)],
        "kinematic_anomaly_score": 0.5,
        "isolated_suspects": [],
    }, "Where is the child now?"),
    ("crowd", {
        "camera_id": "cam1",
        "setting_name": "Entrance & Foyer",
        "occupancy": 15.0,
        "surge_rate": 0.2,
        "compression": 10.0,
        "verdict": "NORMAL",
        "directives": [],
    }, "How is the crowd density?"),
    ("civic", {
        "identifier": "MOTORCYCLE_NODE_77",
        "camera_id": "cam7",
        "setting_name": "Emergency Exit & Loading Dock",
        "compliance_status": "TRAFFIC_INFRACTION_DETECTED",
        "violations": [{"type": "HELMET_GAP_VIOLATION", "details": "Pedestrian 2 missing helmet", "severity": "CRITICAL"}],
    }, "Are there any infractions?"),
]

for name, ctx, query in usecases:
    print(f"\n--- Testing usecase: {name} ---")
    t0 = time.time()
    res = generate_dynamic_usecase_briefing(name, ctx, user_query=query)
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")
    print("Response:")
    print(res)
