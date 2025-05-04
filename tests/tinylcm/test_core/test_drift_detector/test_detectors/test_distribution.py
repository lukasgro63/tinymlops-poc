import pytest

from tinylcm.core.drift_detector.detectors.distribution import DistributionDriftDetector

def test_distribution_detector_process_reference(sample_distribution_data):
    detector = DistributionDriftDetector()
    
    processed = detector._process_reference_data(sample_distribution_data)
    
    assert "class_distribution" in processed
    
    distribution = processed["class_distribution"]
    assert "cat" in distribution
    assert "dog" in distribution
    assert "bird" in distribution
    
    assert sum(distribution.values()) == pytest.approx(1.0)

def test_distribution_detector_calculate_drift(temp_dir, reference_distribution, 
                                              current_distribution):
    detector = DistributionDriftDetector(storage_dir=temp_dir, threshold=0.2)
    
    detector.reference_distribution = reference_distribution
    
    for _ in range(10):
        for cls, prob in current_distribution["class_distribution"].items():
            count = int(prob * 10)
            for _ in range(count):
                detector.current_window.append({"prediction": cls})
    
    result = detector._calculate_drift()
    
    assert "drift_detected" in result
    assert "drift_type" in result
    assert "distance" in result
    assert "reference_distribution" in result
    assert "current_distribution" in result

def test_distribution_detector_no_drift_on_similar(temp_dir, reference_distribution):
    detector = DistributionDriftDetector(storage_dir=temp_dir, threshold=0.2)
    
    detector.reference_distribution = reference_distribution
    
    for _ in range(10):
        for cls, prob in reference_distribution["class_distribution"].items():
            count = int(prob * 10)
            for _ in range(count):
                detector.current_window.append({"prediction": cls})
    
    result = detector._calculate_drift()
    
    assert result["drift_detected"] is False

def test_distribution_detector_drift_on_different(temp_dir, reference_distribution, 
                                                 current_distribution):
    detector = DistributionDriftDetector(storage_dir=temp_dir, threshold=0.15)
    
    detector.reference_distribution = reference_distribution
    
    for _ in range(10):
        for cls, prob in current_distribution["class_distribution"].items():
            count = int(prob * 10)
            for _ in range(count):
                detector.current_window.append({"prediction": cls})
    
    result = detector._calculate_drift()
    
    assert result["drift_detected"] is True