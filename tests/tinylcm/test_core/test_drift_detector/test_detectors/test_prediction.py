import pytest

from tinylcm.core.drift_detector.detectors.prediction import PredictionFrequencyDriftDetector

def test_prediction_detector_process_reference(sample_distribution_data):
    detector = PredictionFrequencyDriftDetector()
    
    processed = detector._process_reference_data(sample_distribution_data)
    
    assert "class_frequencies" in processed
    
    frequencies = processed["class_frequencies"]
    assert "cat" in frequencies
    assert "dog" in frequencies
    assert "bird" in frequencies
    
    assert sum(frequencies.values()) == pytest.approx(1.0)

def test_prediction_detector_calculate_drift(temp_dir, reference_distribution):
    detector = PredictionFrequencyDriftDetector(storage_dir=temp_dir, threshold=0.2)
    
    detector.reference_distribution = {
        "class_frequencies": reference_distribution["class_distribution"]
    }
    
    current_dist = {
        "cat": 0.2,
        "dog": 0.6,
        "bird": 0.15,
        "fish": 0.05
    }
    
    for _ in range(10):
        for cls, freq in current_dist.items():
            count = int(freq * 10)
            for _ in range(count):
                detector.current_window.append({"prediction": cls})
    
    result = detector._calculate_drift()
    
    assert "drift_detected" in result
    assert "drift_type" in result
    assert "distance" in result
    assert "class_frequencies" in result
    
    assert result["drift_detected"] is True

def test_prediction_detector_no_drift_on_similar(temp_dir, reference_distribution):
    detector = PredictionFrequencyDriftDetector(storage_dir=temp_dir, threshold=0.1)
    
    detector.reference_distribution = {
        "class_frequencies": reference_distribution["class_distribution"]
    }
    
    for _ in range(10):
        for cls, freq in reference_distribution["class_distribution"].items():
            count = int(freq * 10)
            for _ in range(count):
                detector.current_window.append({"prediction": cls})
    
    result = detector._calculate_drift()
    
    assert result["drift_detected"] is False

def test_tvd_calculation(temp_dir):
    detector = PredictionFrequencyDriftDetector(storage_dir=temp_dir)
    
    detector.reference_distribution = {
        "class_frequencies": {
            "A": 0.5,
            "B": 0.5
        }
    }
    
    for _ in range(5):
        detector.current_window.append({"prediction": "B"})
    
    result = detector._calculate_drift()
    
    assert result["distance"] == pytest.approx(0.5)