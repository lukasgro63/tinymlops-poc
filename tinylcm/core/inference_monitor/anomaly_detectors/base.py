from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from tinylcm.interfaces.monitoring import AnomalyDetector as IAnomalyDetector
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)

class AnomalyDetector(IAnomalyDetector):

    @abstractmethod
    def check_for_anomalies(
        self,
        record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:

        pass