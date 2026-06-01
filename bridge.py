import time
import os
import requests
from typing import List, Dict

class URGISStreamBridge:
    def __init__(self, target_url: str = "http://127.0.0.1:8000/api/v1/telemetry"):
        self.target_url = target_url
        self.session = requests.Session() 
        print(f"[BRIDGE INIT] Ingestion bridge bound to API gateway: {self.target_url}")

    def dispatch_packet(self, payload: Dict) -> bool:
        try:
            response = self.session.post(self.target_url, json=payload, timeout=0.5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"[BRIDGE ERROR] Failed to connect to orchestrator API: {e}")
            return False

    def stream_mock_wildtrack_dataset(self, total_frames: int = 40):
        print(f"\n[STREAM] Initializing active telemetry stream loop ({total_frames} frames at 30 FPS)...")
        time_base = time.time()

        for frame_idx in range(total_frames):
            timestamp = time_base + (frame_idx * 0.033)
            
            active_entities = [
                {
                    "camera_id": "cam1",
                    "entity_id": "WILDTRACK_PED_42",
                    "x_meters": float(0.5 + (frame_idx * 0.08)),
                    "y_meters": float(1.2 + (frame_idx * 0.11)),
                    "frame_timestamp": timestamp
                },
                {
                    "camera_id": "cam1",
                    "entity_id": "WILDTRACK_PED_99",
                    "x_meters": float(4.5 - (frame_idx * 0.06)),
                    "y_meters": float(5.0 - (frame_idx * 0.07)),
                    "frame_timestamp": timestamp
                },
                {
                    "camera_id": "cam1",
                    "entity_id": "19",
                    "x_meters": float(2.0 + (frame_idx * 0.05)),
                    "y_meters": float(3.0 + (frame_idx * 0.06)),
                    "frame_timestamp": timestamp
                }
            ]

            for packet in active_entities:
                success = self.dispatch_packet(packet)
                if not success:
                    print("[BRIDGE] Stream paused due to connection errors.")
                    return

            print(f"  ↳ Stream Injected -> Frame {frame_idx:02d} | Packets Dispatched: {len(active_entities)} | TS: {timestamp:.2f}")
            
            time.sleep(0.033)

        print("\n[STREAM] Telemetry stream run completed successfully.")

    def stream_raw_wildtrack_csv(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"[ERROR] Target data path not found: {file_path}")
            return
            
        print(f"[STREAM] Parsing raw data rows from source file: {file_path}")
        pass

if __name__ == "__main__":
    bridge = URGISStreamBridge()
    try:
        while True:
            bridge.stream_mock_wildtrack_dataset(total_frames=100)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[BRIDGE] Stopped by analyst.")