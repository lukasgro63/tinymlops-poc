from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tinylcm.core.drift_detector.base import DriftDetector


class DistributionDriftDetector(DriftDetector):
    def _process_reference_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        distribution = {}
        
        if "predictions" in data and isinstance(data["predictions"], list):
            predictions = data["predictions"]
            counter = Counter(predictions)
            total = len(predictions)
            
            if total > 0:
                distribution["class_distribution"] = {
                    cls: count / total for cls, count in counter.items()
                }
        
        return distribution
    
    def _calculate_drift(self) -> Dict[str, Any]:
        result = super()._calculate_drift()
        
        if self.reference_distribution is None or "class_distribution" not in self.reference_distribution:
            return {**result, "error": "Invalid reference distribution"}
        
        predictions = [r.get("prediction") for r in self.current_window if "prediction" in r]
        counter = Counter(predictions)
        total = len(predictions)
        
        if total == 0:
            return {**result, "error": "No predictions in current window"}
        
        current_distribution = {cls: count / total for cls, count in counter.items()}
        
        reference_dist = self.reference_distribution["class_distribution"]
        
        tvd = 0.0
        all_categories = set(reference_dist.keys()) | set(current_distribution.keys())
        
        for category in all_categories:
            ref_prob = reference_dist.get(category, 0.0)
            cur_prob = current_distribution.get(category, 0.0)
            tvd += abs(ref_prob - cur_prob)
        
        tvd = tvd / 2.0
        
        drift_detected = bool(tvd > self.threshold)
        
        self.logger.debug(f"TVD between distributions: {tvd}, threshold: {self.threshold}")
        self.logger.debug(f"Reference distribution: {reference_dist}")
        self.logger.debug(f"Current distribution: {current_distribution}")
        
        result.update({
            "drift_detected": drift_detected,
            "drift_type": "prediction_distribution",
            "distance": float(tvd),
            "similarity_score": float(1 - tvd),
            "threshold": self.threshold,
            "reference_distribution": reference_dist,
            "current_distribution": current_distribution
        })
        
        return result