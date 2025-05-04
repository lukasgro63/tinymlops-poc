from typing import Any, Dict, List, Tuple

from tinylcm.core.inference_monitor.anomaly_detectors.base import AnomalyDetector
from tinylcm.utils.logging import setup_logger

class CompositeAnomalyDetector(AnomalyDetector):
    def __init__(self, detectors: List[AnomalyDetector]):
        self.detectors = detectors
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

    def check_for_anomalies(
        self,
        record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:

        is_anomaly = False
        all_reasons = []

        for detector in self.detectors:
            detector_result, detector_reasons = detector.check_for_anomalies(record, context)

            if detector_result:
                is_anomaly = True
                all_reasons.extend(detector_reasons)

        return is_anomaly, all_reasons