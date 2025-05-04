import collections
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Generic, List, Optional, Protocol, Sequence, Tuple, TypeVar, Union, cast

import numpy as np

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class MetricsProvider(Protocol):
    def get_metrics(self) -> Dict[str, Any]:
        ...

class MetricsConsumer(Protocol):
    def consume_metrics(self, metrics: Dict[str, Any]) -> None:
        ...

class MetricsCalculator:
    @staticmethod
    def accuracy(predictions: Sequence[Any], ground_truth: Sequence[Any]) -> float:
        if len(predictions) == 0 or len(ground_truth) == 0:
            raise ValueError("Empty input sequences")
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Length mismatch: predictions ({len(predictions)}) vs ground_truth ({len(ground_truth)})")
        correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
        return correct / len(predictions)

    @staticmethod
    def confusion_matrix(predictions: Sequence[Any], ground_truth: Sequence[Any]) -> Dict[Any, Dict[Any, int]]:
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Length mismatch: predictions ({len(predictions)}) vs ground_truth ({len(ground_truth)})")
        classes = sorted(set(list(predictions) + list(ground_truth)))
        conf_matrix: Dict[Any, Dict[Any, int]] = {
            true_class: {pred_class: 0 for pred_class in classes}
            for true_class in classes
        }
        for pred, gt in zip(predictions, ground_truth):
            conf_matrix[pred][gt] += 1
        return conf_matrix

    @staticmethod
    def precision_recall_f1(confusion_matrix: Dict[Any, Dict[Any, int]]) -> Dict[Any, Dict[str, float]]:
        metrics: Dict[Any, Dict[str, float]] = {}
        for true_class in confusion_matrix:
            tp = confusion_matrix[true_class][true_class]
            fp = sum(confusion_matrix[other_class][true_class] for other_class in confusion_matrix if other_class != true_class)
            fn = sum(confusion_matrix[true_class][other_class] for other_class in confusion_matrix if other_class != true_class)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics[true_class] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        return metrics

    @staticmethod
    def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
        if not latencies:
            return {
                "min_ms": 0.0,
                "max_ms": 0.0,
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "std_ms": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "std": 0.0
            }
        latencies_array = np.array(latencies)
        min_val = float(np.min(latencies_array))
        max_val = float(np.max(latencies_array))
        mean_val = float(np.mean(latencies_array))
        median_val = float(np.median(latencies_array))
        p95_val = float(np.percentile(latencies_array, 95))
        p99_val = float(np.percentile(latencies_array, 99))
        std_val = float(np.std(latencies_array))
        stats = {
            "min_ms": min_val,
            "max_ms": max_val,
            "mean_ms": mean_val,
            "median_ms": median_val,
            "p95_ms": p95_val,
            "p99_ms": p99_val,
            "std_ms": std_val,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "median": median_val,
            "p95": p95_val,
            "p99": p99_val,
            "std": std_val
        }
        return stats

    @staticmethod
    def calculate_confidence_stats(confidences: List[float]) -> Dict[str, float]:
        if not confidences:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p5": 0.0,
                "std": 0.0
            }
        confidences_array = np.array(confidences)
        stats = {
            "min": float(np.min(confidences_array)),
            "max": float(np.max(confidences_array)),
            "mean": float(np.mean(confidences_array)),
            "median": float(np.median(confidences_array)),
            "p5": float(np.percentile(confidences_array, 5)),
            "std": float(np.std(confidences_array))
        }
        return stats

    @staticmethod
    def distribution_similarity(dist1: Dict[Any, float], dist2: Dict[Any, float]) -> float:
        all_keys = set(dist1.keys()) | set(dist2.keys())
        dist1_complete = {k: dist1.get(k, 0.0) for k in all_keys}
        dist2_complete = {k: dist2.get(k, 0.0) for k in all_keys}
        labels = sorted(all_keys)
        p = np.array([dist1_complete[k] for k in labels])
        q = np.array([dist2_complete[k] for k in labels])
        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q
        m = (p + q) / 2
        divergence = 0.0
        for i in range(len(labels)):
            if p[i] > 0:
                divergence += 0.5 * p[i] * np.log2(p[i] / m[i])
            if q[i] > 0:
                divergence += 0.5 * q[i] * np.log2(q[i] / m[i])
        similarity = 1.0 - min(1.0, divergence)
        return similarity

class Timer:
    def __init__(self):
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

    def start(self) -> None:
        self._start_time = time.time()
        self._stop_time = None

    def stop(self) -> float:
        if self._start_time is None:
            raise ValueError("Timer was not started")
        self._stop_time = time.time()
        return self.elapsed()

    def elapsed(self) -> float:
        if self._start_time is None:
            raise ValueError("Timer was not started")
        end_time = self._stop_time if self._stop_time is not None else time.time()
        return end_time - self._start_time

    def elapsed_ms(self) -> float:
        return self.elapsed() * 1000.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

class MovingAverage:
    def __init__(self, window_size: int = 10):
        self.window_size = max(1, window_size)
        self.values: collections.deque = collections.deque(maxlen=self.window_size)
        self._sum = 0.0

    def add(self, value: float) -> float:
        if len(self.values) == self.window_size:
            self._sum -= self.values[0]
        self.values.append(value)
        self._sum += value
        return self.average()

    def average(self) -> float:
        if not self.values:
            return 0.0
        return self._sum / len(self.values)

    def reset(self) -> None:
        self.values.clear()
        self._sum = 0.0

    def get_values(self) -> List[float]:
        return list(self.values)

    def is_full(self) -> bool:
        return len(self.values) == self.window_size
