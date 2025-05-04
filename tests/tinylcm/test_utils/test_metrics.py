"""Tests for metrics utility functions."""

import time

import pytest

from tinylcm.utils.metrics import MetricsCalculator, MovingAverage, Timer


class TestMetricsCalculator:
    """Test metrics calculation functions."""

    def test_accuracy(self):
        """Test accuracy calculation."""
        predictions = ["cat", "dog", "bird", "cat", "dog"]
        ground_truth = ["cat", "dog", "cat", "cat", "bird"]

        accuracy = MetricsCalculator.accuracy(predictions, ground_truth)

        assert accuracy == 0.6  # 3 correct out of 5

    def test_accuracy_empty_lists_raises_error(self):
        """Test that accuracy with empty lists raises ValueError."""
        with pytest.raises(ValueError):
            MetricsCalculator.accuracy([], [])

    def test_accuracy_mismatched_lengths_raises_error(self):
        """Test that accuracy with mismatched lengths raises ValueError."""
        with pytest.raises(ValueError):
            MetricsCalculator.accuracy(["cat", "dog"], ["cat"])

    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        predictions = ["cat", "dog", "bird", "cat", "dog"]
        ground_truth = ["cat", "dog", "cat", "cat", "bird"]

        conf_matrix = MetricsCalculator.confusion_matrix(predictions, ground_truth)

        assert conf_matrix["cat"]["cat"] == 2  # True positives for cat
        assert conf_matrix["cat"]["bird"] == 0  # Never predicted bird when it was cat
        assert conf_matrix["cat"]["dog"] == 0  # Never predicted dog when it was cat
        assert conf_matrix["bird"]["cat"] == 1  # Predicted cat when it was bird
        assert conf_matrix["dog"]["dog"] == 1  # True positives for dog
        assert conf_matrix["bird"]["bird"] == 0  # Never correctly predicted bird

    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 calculation."""
        # Create a confusion matrix
        conf_matrix = {
            "cat": {"cat": 8, "dog": 2, "bird": 0},
            "dog": {"cat": 3, "dog": 7, "bird": 0},
            "bird": {"cat": 2, "dog": 1, "bird": 4}
        }

        metrics = MetricsCalculator.precision_recall_f1(conf_matrix)

        # Check cat metrics
        assert metrics["cat"]["precision"] == pytest.approx(8 / 13)  # 8 / (8 + 3 + 2)
        assert metrics["cat"]["recall"] == pytest.approx(8 / 10)     # 8 / (8 + 2 + 0)

        # Check F1 calculation
        precision = 8 / 13
        recall = 8 / 10
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        assert metrics["cat"]["f1"] == pytest.approx(expected_f1)

    def test_calculate_latency_stats(self):
        """Test latency statistics calculation."""
        latencies = [5.0, 10.0, 15.0, 20.0, 25.0]

        stats = MetricsCalculator.calculate_latency_stats(latencies)

        assert stats["min_ms"] == 5.0
        assert stats["max_ms"] == 25.0
        assert stats["mean_ms"] == 15.0
        assert stats["median_ms"] == 15.0
        assert "p95_ms" in stats
        assert "p99_ms" in stats
        assert "std_ms" in stats

    def test_calculate_latency_stats_empty_list(self):
        """Test latency statistics with empty input."""
        stats = MetricsCalculator.calculate_latency_stats([])

        assert stats["min_ms"] == 0
        assert stats["max_ms"] == 0
        assert stats["mean_ms"] == 0

    def test_calculate_confidence_stats(self):
        """Test confidence statistics calculation."""
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]

        stats = MetricsCalculator.calculate_confidence_stats(confidences)

        assert stats["min"] == 0.1
        assert stats["max"] == 0.9
        assert stats["mean"] == 0.5
        assert stats["median"] == 0.5
        assert "p5" in stats  # Low confidence threshold
        assert "std" in stats

    def test_distribution_similarity(self):
        """Test distribution similarity calculation."""
        dist1 = {"cat": 0.5, "dog": 0.3, "bird": 0.2}
        dist2 = {"cat": 0.5, "dog": 0.3, "bird": 0.2}
        identical_similarity = MetricsCalculator.distribution_similarity(dist1, dist2)
        assert identical_similarity == pytest.approx(1.0)

        dist3 = {"cat": 0.1, "dog": 0.1, "bird": 0.8}
        different_similarity = MetricsCalculator.distribution_similarity(dist1, dist3)
        assert different_similarity < 0.8  # Should be significantly lower than 1.0


class TestTimer:
    """Test the Timer utility."""

    def test_timer_measures_elapsed_time(self):
        """Test that Timer measures elapsed time correctly."""
        timer = Timer()
        timer.start()
        time.sleep(0.1)  # Sleep for 100ms
        elapsed = timer.stop()

        assert elapsed >= 0.095  # Allow for slight timing variations
        assert elapsed <= 0.15  # Upper bound for timing variations

    def test_timer_elapsed_without_stop(self):
        """Test that elapsed() works without calling stop()."""
        timer = Timer()
        timer.start()
        time.sleep(0.1)  # Sleep for 100ms
        elapsed = timer.elapsed()

        assert elapsed >= 0.095
        assert elapsed <= 0.15

    def test_timer_elapsed_ms(self):
        """Test that elapsed_ms() returns milliseconds."""
        timer = Timer()
        timer.start()
        time.sleep(0.1)  # Sleep for 100ms
        elapsed_ms = timer.elapsed_ms()

        assert elapsed_ms >= 95  # At least 95ms
        assert elapsed_ms <= 150  # At most 150ms

    def test_timer_raises_if_not_started(self):
        """Test that Timer raises ValueError if not started."""
        timer = Timer()

        with pytest.raises(ValueError):
            timer.elapsed()

        with pytest.raises(ValueError):
            timer.stop()


class TestMovingAverage:
    """Test the MovingAverage utility."""

    def test_moving_average_initial_empty(self):
        """Test that moving average starts empty."""
        ma = MovingAverage(window_size=3)
        assert ma.average() == 0

    def test_moving_average_adds_values(self):
        """Test that moving average correctly adds values."""
        ma = MovingAverage(window_size=3)

        assert ma.add(10) == 10  # First value
        assert ma.add(20) == 15  # Average of 10, 20
        assert ma.add(30) == 20  # Average of 10, 20, 30

        # Window should be full now, oldest value gets dropped
        assert ma.add(40) == 30  # Average of 20, 30, 40

    def test_moving_average_window_size(self):
        """Test that moving average respects window size."""
        ma = MovingAverage(window_size=2)

        ma.add(10)
        ma.add(20)
        assert ma.average() == 15

        # Adding third value should drop the first
        ma.add(30)
        assert ma.average() == 25  # Average of 20, 30

    def test_moving_average_reset(self):
        """Test that reset clears the moving average."""
        ma = MovingAverage(window_size=3)

        ma.add(10)
        ma.add(20)
        assert ma.average() == 15

        ma.reset()
        assert ma.average() == 0

        ma.add(30)
        assert ma.average() == 30  # Only one value after reset
