from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tinylcm.core.drift_detector.base import DriftDetector


class FeatureStatisticsDriftDetector(DriftDetector):
    def _process_reference_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        stats = {}
        
        if "features" in data and isinstance(data["features"], list):
            for feature in data["features"]:
                if "name" in feature and "values" in feature and isinstance(feature["values"], list):
                    name = feature["name"]
                    values = feature["values"]
                    
                    if values:
                        values_array = np.array(values)
                        stats[name] = {
                            "mean": float(np.mean(values_array)),
                            "std": float(np.std(values_array)),
                            "min": float(np.min(values_array)),
                            "max": float(np.max(values_array)),
                            "median": float(np.median(values_array))
                        }
        
        return {"feature_stats": stats}
    
    def _calculate_drift(self) -> Dict[str, Any]:
        result = super()._calculate_drift()
        
        if self.reference_distribution is None or "feature_stats" not in self.reference_distribution:
            return {**result, "error": "Invalid reference distribution"}
        
        feature_values = defaultdict(list)
        
        for record in self.current_window:
            if "features" in record and isinstance(record["features"], dict):
                for name, value in record["features"].items():
                    if isinstance(value, (int, float)):
                        feature_values[name].append(value)
        
        current_stats = {}
        for name, values in feature_values.items():
            if values:
                values_array = np.array(values)
                current_stats[name] = {
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "median": float(np.median(values_array))
                }
        
        reference_stats = self.reference_distribution["feature_stats"]
        feature_drifts = {}
        any_drift = False
        
        for name in set(reference_stats.keys()) & set(current_stats.keys()):
            ref_stat = reference_stats[name]
            cur_stat = current_stats[name]
            
            if ref_stat["std"] > 0:
                z_score = abs(cur_stat["mean"] - ref_stat["mean"]) / ref_stat["std"]
                
                feature_drift = bool(z_score > 2.0)
                
                feature_drifts[name] = {
                    "drift_detected": feature_drift,
                    "z_score": float(z_score),
                    "reference": ref_stat,
                    "current": cur_stat
                }
                
                if feature_drift:
                    any_drift = True
        
        result.update({
            "drift_detected": bool(any_drift),
            "drift_type": "feature_statistics",
            "feature_drifts": feature_drifts,
            "threshold": self.threshold
        })
        
        return result