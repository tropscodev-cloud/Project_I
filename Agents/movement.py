import time
import numpy as np
import torch
from typing import List, Tuple, Dict
from chronos import BaseChronosPipeline

class ZeroShotMovementAgent:
    def __init__(self, prediction_seconds: int = 5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prediction_steps = prediction_seconds
        
        print("[INFO] Loading zero-shot Amazon Chronos-Bolt-Tiny pipeline...")
        self.pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",
            device_map=self.device,
            dtype=torch.float32
        )
        print(f"[INFO] Zero-shot core running actively on: '{self.device}'")

    def forecast_track(self, entity_id: str, sanitized_history: List[Tuple[float, float]]) -> Dict:
        if len(sanitized_history) < 8:
            return {
                "entity_id": entity_id,
                "status": "PENDING",
                "reason": f"Gathering context memory. Have {len(sanitized_history)}/8 tokens."
            }
            
        start_time = time.time()
        
        matrix = np.array(sanitized_history, dtype=np.float32)
        
        x_timeline = torch.tensor(matrix[:, 0], dtype=torch.float32)
        y_timeline = torch.tensor(matrix[:, 1], dtype=torch.float32)
        
        context_batch = torch.stack([x_timeline, y_timeline], dim=0)
        
        with torch.no_grad():
            forecasts = self.pipeline.predict(context_batch, prediction_length=self.prediction_steps)
            
        median_forecasts = torch.median(forecasts, dim=1).values
        pred_x = median_forecasts[0].cpu().numpy().tolist()
        pred_y = median_forecasts[1].cpu().numpy().tolist()
        
        predicted_path = [
            (round(x, 3), round(y, 3)) for x, y in zip(pred_x, pred_y)
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "entity_id": entity_id,
            "status": "SUCCESS",
            "latency_ms": round(latency_ms, 2),
            "predicted_trajectory": predicted_path
        }

if __name__ == "__main__":
    agent = ZeroShotMovementAgent()
    
    wildtrack_mock_stream = [
        (0.5, 1.2), (0.7, 1.5), (0.9, 1.8), (1.1, 2.1),
        (1.3, 2.4), (1.5, 2.7), (1.7, 3.0), (1.9, 3.3)
    ]
    
    print("\n[EXEC] Dispatched historical tracking matrix to Zero-Shot Agent...")
    result = agent.forecast_track(entity_id="WILDTRACK_PED_42", sanitized_history=wildtrack_mock_stream)
    
    print("\n" + "="*60)
    print("               ZERO-SHOT PREDICTION AGENT OUTPUT             ")
    print("="*60)
    print(f"Target Identity : {result['entity_id']}")
    print(f"Agent Status    : {result['status']}")
    print(f"Inference Time  : {result.get('latency_ms')} ms")
    print("-"*60)
    print("Projected Future Trajectory Path (Next 5 Steps):")
    for step, position in enumerate(result['predicted_trajectory'], start=1):
        print(f"  ↳ Step +{step} -> Projected Position: X={position[0]}m, Y={position[1]}m")
    print("="*60)