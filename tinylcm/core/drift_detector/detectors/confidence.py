from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tinylcm.core.drift_detector.base import DriftDetector


class ConfidenceDriftDetector(DriftDetector):
    def _process_reference_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        stats = {}
        
        if "confidences" in data and isinstance(data["confidences"], list):
            confidences = data["confidences"]
            
            if confidences:
                conf_array = np.array(confidences)
                stats["confidence_stats"] = {
                    "mean": float(np.mean(conf_array)),
                    "std": float(np.std(conf_array)),
                    "min": float(np.min(conf_array)),
                    "max": float(np.max(conf_array)),
                    "median": float(np.median(conf_array))
                }
        
        return stats
    
    def _calculate_drift(self) -> Dict[str, Any]:
        result = super()._calculate_drift()
        
        if self.reference_distribution is None or "confidence_stats" not in self.reference_distribution:
            return {**result, "error": "Invalid reference distribution"}
        
        confidences = [r.get("confidence") for r in self.current_window if "confidence" in r]
        
        if not confidences:
            return {**result, "error": "No confidence values in current window"}
        
        conf_array = np.array(confidences)
        current_stats = {
            "mean": float(np.mean(conf_array)),
            "std": float(np.std(conf_array)),
            "min": float(np.min(conf_array)),
            "max": float(np.max(conf_array)),
            "median": float(np.median(conf_array))
        }
        
        reference_stats = self.reference_distribution["confidence_stats"]
        
        ref_mean = reference_stats["mean"]
        cur_mean = current_stats["mean"]
        
        if ref_mean > 0:
            relative_diff = abs(cur_mean - ref_mean) / ref_mean
            drift_detected = bool(relative_diff > self.threshold)
        else:
            drift_detected = bool(abs(cur_mean - ref_mean) > self.threshold)
        
        result.update({
            "drift_detected": drift_detected,
            "drift_type": "confidence_scores",
            "confidence_stats": {
                "reference_mean": float(reference_stats["mean"]),
                "current_mean": float(current_stats["mean"]),
                "relative_difference": float(relative_diff) if ref_mean > 0 else None,
                "reference": reference_stats,
                "current": current_stats
            },
            "threshold": self.threshold
        })
        
        return result