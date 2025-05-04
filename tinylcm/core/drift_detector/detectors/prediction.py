from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tinylcm.core.drift_detector.base import DriftDetector


class PredictionFrequencyDriftDetector(DriftDetector):
    def _process_reference_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        frequencies = {}
        
        if "predictions" in data and isinstance(data["predictions"], list):
            predictions = data["predictions"]
            counter = Counter(predictions)
            total = len(predictions)
            
            if total > 0:
                frequencies["class_frequencies"] = {
                    cls: count / total for cls, count in counter.items()
                }
        
        return frequencies
    
    def _calculate_drift(self) -> Dict[str, Any]:
        result = super()._calculate_drift()
        
        if self.reference_distribution is None or "class_frequencies" not in self.reference_distribution:
            return {**result, "error": "Invalid reference distribution"}
        
        predictions = [r.get("prediction") for r in self.current_window if "prediction" in r]
        counter = Counter(predictions)
        total = len(predictions)
        
        if total == 0:
            return {**result, "error": "No predictions in current window"}
        
        current_frequencies = {cls: count / total for cls, count in counter.items()}
        
        reference_freq = self.reference_distribution["class_frequencies"]
        
        all_classes = set(reference_freq.keys()) | set(current_frequencies.keys())
        
        tvd = 0.0
        for cls in all_classes:
            ref_freq = reference_freq.get(cls, 0.0)
            cur_freq = current_frequencies.get(cls, 0.0)
            tvd += abs(ref_freq - cur_freq)
        
        tvd = tvd / 2.0
        
        drift_detected = bool(tvd > self.threshold)
        
        result.update({
            "drift_detected": drift_detected,
            "drift_type": "prediction_frequency",
            "distance": float(tvd),
            "threshold": self.threshold,
            "class_frequencies": {
                "reference": reference_freq,
                "current": current_frequencies
            }
        })
        
        return result