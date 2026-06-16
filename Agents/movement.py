import time
import numpy as np
import torch
from typing import List, Tuple, Dict
from chronos import BaseChronosPipeline

class ZeroShotMovementAgent:
    def __init__(self, prediction_seconds: int = 5):
        # FIX: Complete auto-detection matrix supporting Apple Silicon (MPS), NVIDIA (CUDA), and fallback (CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self.prediction_steps = prediction_seconds
        
        print(f"[INFO] Initializing Zero-Shot Amazon Chronos Pipeline on device: '{self.device}'")
        self.pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",
            device_map=self.device,
            dtype=torch.float32
        )

    def forecast_track(self, entity_id: str, sanitized_history: List[Tuple[float, float]]) -> Dict:
        if len(sanitized_history) < 8:
            return {
                "entity_id": entity_id,
                "status": "PENDING",
                "reason": f"Gathering sequence token contexts. Have {len(sanitized_history)}/8 vectors."
            }
            
        start_time = time.time()
        
        # Enforce isolated, clean thread execution space
        matrix = np.array(sanitized_history, dtype=np.float32)
        
        # Explicit device allocation matching current application stack
        x_timeline = torch.tensor(matrix[:, 0], dtype=torch.float32, device=self.device)
        y_timeline = torch.tensor(matrix[:, 1], dtype=torch.float32, device=self.device)
        
        context_batch = torch.stack([x_timeline, y_timeline], dim=0)
        
        # Non-deterministic probabilistic inference stream
        with torch.no_grad():
            forecasts = self.pipeline.predict(context_batch, self.prediction_steps)
            
        # Extract median coordinates safely and cleanly detach from graphics processor memory maps
        median_forecasts = torch.median(forecasts, dim=1).values
        pred_x = median_forecasts[0].cpu().numpy().tolist()
        pred_y = median_forecasts[1].cpu().numpy().tolist()
        
        predicted_path = [(round(x, 3), round(y, 3)) for x, y in zip(pred_x, pred_y)]
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "entity_id": entity_id,
            "status": "SUCCESS",
            "latency_ms": round(latency_ms, 2),
            "predicted_trajectory": predicted_path
        }

if __name__ == "__main__":
    agent = ZeroShotMovementAgent()
    mock_history = [(0.5, 1.2), (0.7, 1.5), (0.9, 1.8), (1.1, 2.1), (1.3, 2.4), (1.5, 2.7), (1.7, 3.0), (1.9, 3.3)]
    print(agent.forecast_track("WILDTRACK_PED_42", mock_history))