from typing import Any, Dict, List, Tuple

from tinylcm.core.inference_monitor.anomaly_detectors.base import AnomalyDetector
from tinylcm.utils.logging import setup_logger

class ThresholdAnomalyDetector(AnomalyDetector):
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        latency_threshold_ms: float = 100.0
    ):
        self.confidence_threshold = confidence_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

    def check_for_anomalies(
        self,
        record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        reasons = []
        if (
            "confidence" in record and
            record["confidence"] is not None and
            record["confidence"] < self.confidence_threshold
        ):
            reasons.append(f"Low confidence: {record['confidence']:.4f} < {self.confidence_threshold:.4f}")
        if (
            "latency_ms" in record and
            record["latency_ms"] is not None and
            record["latency_ms"] > self.latency_threshold_ms
        ):
            reasons.append(f"High latency: {record['latency_ms']:.2f}ms > {self.latency_threshold_ms:.2f}ms")
        if (
            "ground_truth" in record and
            record["ground_truth"] is not None and
            "prediction" in record and
            record["prediction"] != record["ground_truth"]
        ):
            reasons.append(f"Incorrect prediction: {record['prediction']} != {record['ground_truth']}")
        return bool(reasons), reasons