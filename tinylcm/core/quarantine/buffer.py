from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
import numpy as np
import time
import threading
import queue
import json
import os
import uuid
from pathlib import Path

from tinylcm.utils.logging import setup_logger
from tinylcm.utils.file_utils import ensure_directory_exists
from tinylcm.core.data_structures import FeatureSample

logger = setup_logger(__name__)


class QuarantineStrategy(Enum):
    """Strategies for determining which samples to quarantine."""
    ALL_DRIFT_SAMPLES = auto()  # Quarantine all samples that cause drift detection
    HIGH_UNCERTAINTY = auto()   # Quarantine samples with low confidence
    OUTLIERS = auto()           # Quarantine feature outliers
    DISTRIBUTION_SHIFT = auto() # Quarantine samples indicating distribution shift
    CUSTOM = auto()             # Custom strategy defined by user-provided function


class QuarantineStatus(Enum):
    """Status of a quarantined sample."""
    PENDING = auto()            # Awaiting review or validation
    VALIDATED = auto()          # Validated with confirmed label
    PSEUDO_LABELED = auto()     # Assigned a pseudo-label by heuristic
    REJECTED = auto()           # Rejected as not useful or noisy
    ADDED_TO_MODEL = auto()     # Added to the model's training data


@dataclass
class QuarantinedSample:
    """A sample placed in quarantine for review or validation.
    
    This represents data that the autonomous drift detectors flagged as
    potentially interesting, unusual, or indicative of drift.
    """
    # Core data
    sample_id: str
    features: np.ndarray
    prediction: Any
    confidence: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    
    # Drift information
    drift_type: str = "unknown"  # Type of drift that triggered quarantine
    drift_score: float = 0.0     # Numeric score indicating drift magnitude
    drift_detector: str = ""     # Name of detector that flagged this sample
    
    # Validation information
    status: QuarantineStatus = QuarantineStatus.PENDING
    true_label: Optional[Any] = None  # Validated true label (if available)
    pseudo_label: Optional[Any] = None  # Heuristically assigned label
    pseudo_label_confidence: float = 0.0  # Confidence in pseudo-label
    
    # Additional data
    raw_data: Optional[Any] = None  # Original input data (e.g., image) if available
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sample_id": self.sample_id,
            "features": self.features.tolist() if isinstance(self.features, np.ndarray) else self.features,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "drift_type": self.drift_type,
            "drift_score": self.drift_score,
            "drift_detector": self.drift_detector,
            "status": self.status.name,
            "true_label": self.true_label,
            "pseudo_label": self.pseudo_label,
            "pseudo_label_confidence": self.pseudo_label_confidence,
            # Do not include raw_data by default to save space
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuarantinedSample':
        """Create a QuarantinedSample from a dictionary."""
        # Convert features back to numpy array
        features = data["features"]
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Convert status string to enum
        status_str = data.get("status", "PENDING")
        status = getattr(QuarantineStatus, status_str)
        
        return cls(
            sample_id=data["sample_id"],
            features=features,
            prediction=data["prediction"],
            confidence=data.get("confidence"),
            timestamp=data.get("timestamp", time.time()),
            drift_type=data.get("drift_type", "unknown"),
            drift_score=data.get("drift_score", 0.0),
            drift_detector=data.get("drift_detector", ""),
            status=status,
            true_label=data.get("true_label"),
            pseudo_label=data.get("pseudo_label"),
            pseudo_label_confidence=data.get("pseudo_label_confidence", 0.0),
            metadata=data.get("metadata", {})
        )
    
    def create_feature_sample(self) -> FeatureSample:
        """Create a FeatureSample from this quarantined sample."""
        # Use true label if available, otherwise pseudo label
        label = self.true_label if self.true_label is not None else self.pseudo_label
        
        return FeatureSample(
            features=self.features,
            label=label,
            prediction=self.prediction,
            timestamp=self.timestamp,
            sample_id=self.sample_id,
            metadata={
                "quarantine_info": {
                    "drift_type": self.drift_type,
                    "drift_score": self.drift_score,
                    "drift_detector": self.drift_detector,
                    "status": self.status.name,
                    "pseudo_label": self.pseudo_label,
                    "pseudo_label_confidence": self.pseudo_label_confidence
                },
                **self.metadata
            }
        )


class QuarantineBuffer:
    """Buffer for storing and managing samples that need validation or review.
    
    The QuarantineBuffer collects samples flagged by autonomous drift detectors 
    for further analysis, validation, or heuristic labeling. It provides mechanisms 
    for managing these samples, including:
    
    1. Adding new samples to quarantine
    2. Retrieving samples for review
    3. Updating sample status based on validation
    4. Exporting samples for external validation
    5. Applying heuristic pseudo-labels
    6. Selecting samples for model adaptation
    
    The buffer maintains persistence through non-blocking I/O and can be
    synchronized with external systems for validation.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        storage_dir: str = "./quarantine",
        auto_persist: bool = True,
        persist_interval: int = 10,
        worker_thread_daemon: bool = True,
        quarantine_strategy: QuarantineStrategy = QuarantineStrategy.ALL_DRIFT_SAMPLES,
        confidence_threshold: float = 0.6,
        outlier_threshold: float = 3.0,
        custom_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
    ):
        """Initialize the quarantine buffer.
        
        Args:
            max_size: Maximum number of samples to keep in buffer
            storage_dir: Directory for persisting quarantined samples
            auto_persist: Whether to automatically persist buffer state
            persist_interval: Number of changes before persisting
            worker_thread_daemon: Whether worker thread should be daemon
            quarantine_strategy: Strategy for determining samples to quarantine
            confidence_threshold: Threshold for HIGH_UNCERTAINTY strategy
            outlier_threshold: Threshold for OUTLIERS strategy
            custom_filter: Custom function for CUSTOM strategy
        """
        self.max_size = max_size
        self.storage_dir = Path(storage_dir)
        self.auto_persist = auto_persist
        self.persist_interval = persist_interval
        
        # Create storage directory
        ensure_directory_exists(self.storage_dir)
        
        # Initialize samples buffer, sorted by timestamp (newest first)
        self.samples: Dict[str, QuarantinedSample] = {}
        self.samples_by_status: Dict[QuarantineStatus, Set[str]] = {
            status: set() for status in QuarantineStatus
        }
        
        # Tracking fields
        self.change_count = 0
        self.last_persist_time = time.time()
        self.session_id = str(uuid.uuid4())
        
        # Quarantine strategy configuration
        self.quarantine_strategy = quarantine_strategy
        self.confidence_threshold = confidence_threshold
        self.outlier_threshold = outlier_threshold
        self.custom_filter = custom_filter
        
        # Worker thread for non-blocking I/O
        self._task_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=worker_thread_daemon
        )
        self._worker_thread.start()
        logger.debug(f"QuarantineBuffer initialized with strategy: {quarantine_strategy.name}")
    
    def should_quarantine(self, record: Dict[str, Any], drift_info: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if a sample should be quarantined based on the strategy.
        
        Args:
            record: The inference record containing prediction, confidence, etc.
            drift_info: Optional information from a drift detector
            
        Returns:
            True if the sample should be quarantined, False otherwise
        """
        if self.quarantine_strategy == QuarantineStrategy.ALL_DRIFT_SAMPLES:
            # Quarantine if drift was detected
            return drift_info is not None
        
        elif self.quarantine_strategy == QuarantineStrategy.HIGH_UNCERTAINTY:
            # Quarantine if confidence is below threshold
            confidence = record.get("confidence")
            if confidence is not None and confidence < self.confidence_threshold:
                return True
            return False
        
        elif self.quarantine_strategy == QuarantineStrategy.OUTLIERS:
            # This requires feature information
            features = record.get("features")
            if features is None:
                return False
            
            # Simplistic outlier detection based on feature magnitude
            # In practice, you would use more sophisticated methods
            if isinstance(features, np.ndarray):
                feature_norm = np.linalg.norm(features)
                return feature_norm > self.outlier_threshold
            return False
        
        elif self.quarantine_strategy == QuarantineStrategy.DISTRIBUTION_SHIFT:
            # Quarantine if distribution drift was detected
            if drift_info and drift_info.get("detector") == "PredictionDistributionMonitor":
                return True
            return False
        
        elif self.quarantine_strategy == QuarantineStrategy.CUSTOM and self.custom_filter:
            # Use custom filter function
            return self.custom_filter(record)
        
        # Default: don't quarantine
        return False
    
    def add_sample(
        self,
        features: np.ndarray,
        prediction: Any,
        drift_info: Dict[str, Any],
        confidence: Optional[float] = None,
        raw_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sample_id: Optional[str] = None
    ) -> str:
        """Add a new sample to the quarantine buffer.
        
        Args:
            features: Feature vector
            prediction: Model prediction
            drift_info: Information about the drift that triggered quarantine
            confidence: Optional confidence score
            raw_data: Optional raw input data
            metadata: Optional additional metadata
            sample_id: Optional unique ID (generated if not provided)
            
        Returns:
            ID of the quarantined sample
        """
        # Generate sample ID if not provided
        if sample_id is None:
            sample_id = str(uuid.uuid4())
        
        # Create quarantined sample
        sample = QuarantinedSample(
            sample_id=sample_id,
            features=features,
            prediction=prediction,
            confidence=confidence,
            drift_type=drift_info.get("metric", "unknown"),
            drift_score=drift_info.get("current_value", 0.0),
            drift_detector=drift_info.get("detector", ""),
            raw_data=raw_data,
            metadata=metadata or {}
        )
        
        # Add to buffer
        self.samples[sample_id] = sample
        self.samples_by_status[QuarantineStatus.PENDING].add(sample_id)
        
        # Enforce max size
        self._enforce_max_size()
        
        # Increment change counter
        self.change_count += 1
        
        # Auto-persist if needed
        if self.auto_persist and self.change_count >= self.persist_interval:
            self.persist_async()
        
        logger.debug(f"Added sample {sample_id} to quarantine (drift: {drift_info.get('detector')})")
        return sample_id
    
    def _enforce_max_size(self) -> None:
        """Enforce maximum buffer size by removing oldest samples if needed."""
        if len(self.samples) <= self.max_size:
            return
        
        # Sort samples by timestamp (oldest first)
        sorted_samples = sorted(
            self.samples.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Remove oldest samples
        samples_to_remove = sorted_samples[:(len(self.samples) - self.max_size)]
        for sample_id, sample in samples_to_remove:
            self._remove_sample(sample_id)
    
    def _remove_sample(self, sample_id: str) -> None:
        """Remove a sample from the buffer."""
        if sample_id in self.samples:
            # Remove from status sets
            sample = self.samples[sample_id]
            self.samples_by_status[sample.status].discard(sample_id)
            
            # Remove from main dict
            del self.samples[sample_id]
            
            logger.debug(f"Removed sample {sample_id} from quarantine")
    
    def get_sample(self, sample_id: str) -> Optional[QuarantinedSample]:
        """Get a sample by ID.
        
        Args:
            sample_id: ID of the sample to retrieve
            
        Returns:
            The sample or None if not found
        """
        return self.samples.get(sample_id)
    
    def get_samples_by_status(self, status: QuarantineStatus) -> List[QuarantinedSample]:
        """Get all samples with a given status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of samples with the specified status
        """
        sample_ids = self.samples_by_status[status]
        return [self.samples[sample_id] for sample_id in sample_ids]
    
    def update_sample_status(
        self,
        sample_id: str,
        status: QuarantineStatus,
        true_label: Optional[Any] = None,
        pseudo_label: Optional[Any] = None,
        pseudo_label_confidence: Optional[float] = None
    ) -> bool:
        """Update the status of a quarantined sample.
        
        Args:
            sample_id: ID of the sample to update
            status: New status
            true_label: Optional validated true label
            pseudo_label: Optional heuristically assigned label
            pseudo_label_confidence: Optional confidence in the pseudo-label
            
        Returns:
            True if successful, False if sample not found
        """
        sample = self.samples.get(sample_id)
        if sample is None:
            logger.warning(f"Sample {sample_id} not found in quarantine")
            return False
        
        # Update status tracking
        old_status = sample.status
        self.samples_by_status[old_status].discard(sample_id)
        self.samples_by_status[status].add(sample_id)
        
        # Update sample
        sample.status = status
        
        if true_label is not None:
            sample.true_label = true_label
        
        if pseudo_label is not None:
            sample.pseudo_label = pseudo_label
        
        if pseudo_label_confidence is not None:
            sample.pseudo_label_confidence = pseudo_label_confidence
        
        # Increment change counter
        self.change_count += 1
        
        # Auto-persist if needed
        if self.auto_persist and self.change_count >= self.persist_interval:
            self.persist_async()
        
        logger.debug(f"Updated sample {sample_id} status to {status.name}")
        return True
    
    def persist(self) -> None:
        """Persist the buffer state to disk (blocking)."""
        try:
            # Create samples file
            samples_file = self.storage_dir / f"quarantine_samples_{self.session_id}.json"
            
            # Convert samples to dict for serialization
            samples_dict = {
                sample_id: sample.to_dict()
                for sample_id, sample in self.samples.items()
            }
            
            # Save to file
            with open(samples_file, 'w') as f:
                json.dump(samples_dict, f, indent=2)
            
            # Reset change counter
            self.change_count = 0
            self.last_persist_time = time.time()
            
            logger.debug(f"Persisted quarantine buffer to {samples_file}")
        except Exception as e:
            logger.error(f"Error persisting quarantine buffer: {str(e)}")
    
    def persist_async(self) -> None:
        """Persist the buffer state to disk asynchronously."""
        # Clone current samples for the worker thread
        samples_copy = {
            sample_id: sample.to_dict()
            for sample_id, sample in self.samples.items()
        }
        
        # Queue the task
        self._task_queue.put(("persist", samples_copy))
    
    def load(self, file_path: Union[str, Path]) -> int:
        """Load buffer state from a file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Number of samples loaded
        """
        try:
            # Clear current samples
            self.samples.clear()
            for status in QuarantineStatus:
                self.samples_by_status[status].clear()
            
            # Load from file
            with open(file_path, 'r') as f:
                samples_dict = json.load(f)
            
            # Convert to samples
            for sample_id, sample_data in samples_dict.items():
                sample = QuarantinedSample.from_dict(sample_data)
                self.samples[sample_id] = sample
                self.samples_by_status[sample.status].add(sample_id)
            
            logger.info(f"Loaded {len(self.samples)} samples into quarantine from {file_path}")
            return len(self.samples)
        except Exception as e:
            logger.error(f"Error loading quarantine buffer: {str(e)}")
            return 0
    
    def get_validated_samples(self) -> List[QuarantinedSample]:
        """Get all samples that have been validated.
        
        Returns:
            List of validated samples
        """
        return self.get_samples_by_status(QuarantineStatus.VALIDATED)
    
    def get_pseudo_labeled_samples(self, min_confidence: float = 0.0) -> List[QuarantinedSample]:
        """Get all samples with pseudo-labels.
        
        Args:
            min_confidence: Minimum confidence threshold for pseudo-labels
            
        Returns:
            List of pseudo-labeled samples meeting the confidence threshold
        """
        samples = self.get_samples_by_status(QuarantineStatus.PSEUDO_LABELED)
        return [s for s in samples if s.pseudo_label_confidence >= min_confidence]
    
    def get_pending_samples(self) -> List[QuarantinedSample]:
        """Get all pending samples awaiting review.
        
        Returns:
            List of pending samples
        """
        return self.get_samples_by_status(QuarantineStatus.PENDING)
    
    def get_samples_for_sync(self) -> List[Dict[str, Any]]:
        """Get samples that need to be synchronized with the server.
        
        This returns all pending samples that haven't been synced yet.
        
        Returns:
            List of samples as dictionaries
        """
        pending_samples = self.get_pending_samples()
        
        # Filter to samples not yet synced
        return [
            sample.to_dict() for sample in pending_samples
            if not sample.metadata.get("synced", False)
        ]
    
    def mark_as_synced(self, sample_ids: List[str]) -> None:
        """Mark samples as synced with the server.
        
        Args:
            sample_ids: List of sample IDs to mark as synced
        """
        for sample_id in sample_ids:
            sample = self.get_sample(sample_id)
            if sample:
                sample.metadata["synced"] = True
                sample.metadata["sync_timestamp"] = time.time()
        
        # Increment change counter
        self.change_count += 1
        
        # Auto-persist if needed
        if self.auto_persist and self.change_count >= self.persist_interval:
            self.persist_async()
    
    def process_validation_results(self, validation_results: List[Dict[str, Any]]) -> int:
        """Process validation results from the server.
        
        Args:
            validation_results: List of validated samples from server
            
        Returns:
            Number of samples successfully processed
        """
        count = 0
        for result in validation_results:
            sample_id = result.get("sample_id")
            if not sample_id:
                continue
            
            sample = self.get_sample(sample_id)
            if not sample:
                continue
            
            # Update status and labels
            true_label = result.get("true_label")
            status = result.get("status", "VALIDATED")
            status_enum = getattr(QuarantineStatus, status, QuarantineStatus.PENDING)
            
            # Update the sample
            success = self.update_sample_status(
                sample_id=sample_id,
                status=status_enum,
                true_label=true_label
            )
            
            if success:
                count += 1
        
        logger.info(f"Processed {count} validation results")
        return count
    
    def _worker_loop(self) -> None:
        """Worker thread loop for background tasks."""
        while not self._stop_event.is_set():
            try:
                # Get task with timeout
                try:
                    task = self._task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process task
                if task[0] == "persist":
                    # Save samples to disk
                    samples_copy = task[1]
                    
                    try:
                        # Create samples file
                        samples_file = self.storage_dir / f"quarantine_samples_{self.session_id}.json"
                        
                        # Save to file
                        with open(samples_file, 'w') as f:
                            json.dump(samples_copy, f, indent=2)
                        
                        # Reset change counter (safe since we're only modifying atomic values)
                        self.change_count = 0
                        self.last_persist_time = time.time()
                        
                        logger.debug(f"Async persisted quarantine buffer to {samples_file}")
                    except Exception as e:
                        logger.error(f"Error in async persist: {str(e)}")
                
                # Mark task as done
                self._task_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in quarantine worker thread: {str(e)}")
                time.sleep(0.1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the quarantine buffer.
        
        Returns:
            Dictionary of statistics
        """
        # Count samples by status
        status_counts = {
            status.name: len(self.samples_by_status[status])
            for status in QuarantineStatus
        }
        
        # Count drift types
        drift_types = {}
        for sample in self.samples.values():
            drift_type = sample.drift_type
            if drift_type not in drift_types:
                drift_types[drift_type] = 0
            drift_types[drift_type] += 1
        
        # Calculate time statistics
        timestamps = [sample.timestamp for sample in self.samples.values()]
        time_stats = {
            "oldest": min(timestamps) if timestamps else None,
            "newest": max(timestamps) if timestamps else None,
            "time_span": max(timestamps) - min(timestamps) if timestamps else None
        }
        
        # Compile statistics
        stats = {
            "total_samples": len(self.samples),
            "status_counts": status_counts,
            "drift_types": drift_types,
            "time_stats": time_stats,
            "session_id": self.session_id,
            "storage_dir": str(self.storage_dir),
            "max_size": self.max_size,
            "quarantine_strategy": self.quarantine_strategy.name
        }
        
        return stats
    
    def close(self) -> None:
        """Close the buffer and clean up resources."""
        # Signal worker thread to stop
        self._stop_event.set()
        
        # Wait for pending tasks to complete
        if self._task_queue.qsize() > 0:
            logger.info(f"Waiting for {self._task_queue.qsize()} pending tasks to complete")
            self._task_queue.join()
        
        # Final persist
        if self.change_count > 0:
            self.persist()
        
        logger.info("Quarantine buffer closed")