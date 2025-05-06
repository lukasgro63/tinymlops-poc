"""
QuarantineBuffer module for TinyLCM.

This module provides a simple buffer for storing samples that are flagged as potentially 
anomalous by drift detectors. These samples can later be analyzed by heuristic 
adapters for on-device adaptation.
"""

from collections import deque
import time
import numpy as np
from typing import Any, Dict, List, Optional, Union

from tinylcm.utils.logging import setup_logger
from tinylcm.core.data_structures import FeatureSample, QuarantinedSample, QuarantineStatus

logger = setup_logger(__name__)


class QuarantineBuffer:
    """Simple FIFO buffer for storing samples that require further analysis.
    
    The QuarantineBuffer temporarily stores samples that have been flagged by 
    autonomous drift detectors. These samples can then be analyzed by the 
    HeuristicAdapter to identify patterns and generate potential labels.
    
    The buffer has configurable size limits and sample expiration to manage
    resource usage on constrained devices.
    """
    
    def __init__(
        self,
        max_size: int = 100,
        max_age: Optional[float] = 3600.0  # 1 hour in seconds
    ):
        """Initialize the quarantine buffer.
        
        Args:
            max_size: Maximum number of samples to store in the buffer
            max_age: Maximum age of samples in seconds (None for no age limit)
        """
        self.max_size = max_size
        self.max_age = max_age
        
        # The main buffer using a deque for efficient FIFO operations
        self.buffer = deque(maxlen=max_size)
        
        # Statistics
        self.total_added = 0
        self.total_removed = 0
        
        logger.debug(
            f"Initialized QuarantineBuffer with max_size={max_size}, "
            f"max_age={max_age}"
        )
    
    def add(
        self,
        sample: FeatureSample,
        reason: str
    ) -> bool:
        """Add a sample to the quarantine buffer.
        
        Args:
            sample: The FeatureSample to add to the buffer
            reason: The reason for quarantining the sample
            
        Returns:
            True if the sample was added, False otherwise
        """
        # Create entry with timestamp and reason
        entry = {
            'sample': sample,
            'reason': reason,
            'timestamp': time.time(),
            'processed': False
        }
        
        # Add to buffer
        self.buffer.append(entry)
        self.total_added += 1
        
        # Clean expired samples
        self._clean_expired()
        
        logger.debug(
            f"Added sample {sample.sample_id} to quarantine buffer "
            f"(reason: {reason}, buffer size: {len(self.buffer)})"
        )
        return True
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Get all samples from the buffer.
        
        Returns:
            List of buffer entries containing samples, reasons, and timestamps
        """
        # Clean expired samples first
        self._clean_expired()
        
        return list(self.buffer)
    
    def clear(self) -> int:
        """Clear the buffer, removing all samples.
        
        Returns:
            Number of samples removed
        """
        count = len(self.buffer)
        self.buffer.clear()
        self.total_removed += count
        
        logger.debug(f"Cleared quarantine buffer, removed {count} samples")
        return count
    
    def mark_as_processed(self, sample_ids: List[str]) -> int:
        """Mark specific samples as processed.
        
        Args:
            sample_ids: List of sample IDs to mark as processed
            
        Returns:
            Number of samples marked as processed
        """
        # Convert to set for faster lookups
        ids_to_mark = set(sample_ids)
        
        # Keep track of how many we marked
        marked = 0
        
        # Mark samples as processed
        for entry in self.buffer:
            sample = entry['sample']
            if sample.sample_id in ids_to_mark and not entry['processed']:
                entry['processed'] = True
                marked += 1
        
        logger.debug(f"Marked {marked} samples as processed in quarantine buffer")
        return marked
    
    def remove_processed(self) -> int:
        """Remove all processed samples from the buffer.
        
        Returns:
            Number of samples removed
        """
        original_size = len(self.buffer)
        
        # Remove processed samples
        self.buffer = deque(
            [entry for entry in self.buffer if not entry['processed']], 
            maxlen=self.max_size
        )
        
        removed = original_size - len(self.buffer)
        self.total_removed += removed
        
        logger.debug(f"Removed {removed} processed samples from quarantine buffer")
        return removed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the quarantine buffer.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            "current_size": len(self.buffer),
            "max_size": self.max_size,
            "total_added": self.total_added,
            "total_removed": self.total_removed,
            "unprocessed_count": sum(1 for entry in self.buffer if not entry['processed']),
            "processed_count": sum(1 for entry in self.buffer if entry['processed']),
            "reasons": self._count_reasons()
        }
    
    def _count_reasons(self) -> Dict[str, int]:
        """Count samples by quarantine reason.
        
        Returns:
            Dictionary mapping reasons to counts
        """
        reason_counts = {}
        for entry in self.buffer:
            reason = entry['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return reason_counts
    
    def _clean_expired(self) -> int:
        """Remove expired samples from the buffer.
        
        Returns:
            Number of samples removed
        """
        # Skip if no age limit
        if self.max_age is None:
            return 0
            
        current_time = time.time()
        original_size = len(self.buffer)
        
        # Keep only non-expired samples
        self.buffer = deque(
            [entry for entry in self.buffer if (current_time - entry['timestamp']) <= self.max_age], 
            maxlen=self.max_size
        )
        
        removed = original_size - len(self.buffer)
        self.total_removed += removed
        
        if removed > 0:
            logger.debug(f"Removed {removed} expired samples from quarantine buffer")
            
        return removed
        
    def get_samples_for_sync(self) -> List[Dict[str, Any]]:
        """Get samples that need to be synchronized with server.
        
        Returns:
            List of samples ready for server synchronization
        """
        samples_to_sync = []
        
        for entry in self.buffer:
            # Only sync samples that are processed but not yet synced
            if entry.get('processed', False) and not entry.get('synced', False):
                # Create a serializable version of the sample
                sample = entry['sample']
                sample_dict = {
                    'sample_id': sample.sample_id,
                    'features': sample.features.tolist() if isinstance(sample.features, np.ndarray) else sample.features,
                    'prediction': sample.prediction,
                    'timestamp': entry['timestamp'],
                    'reason': entry['reason']
                }
                samples_to_sync.append(sample_dict)
                
        return samples_to_sync
        
    def mark_as_synced(self, sample_ids: List[str]) -> int:
        """Mark samples as synchronized with server.
        
        Args:
            sample_ids: List of sample IDs to mark as synced
            
        Returns:
            Number of samples marked as synced
        """
        # Convert to set for faster lookups
        ids_to_mark = set(sample_ids)
        
        # Keep track of how many we marked
        marked = 0
        
        # Mark samples as synced
        for entry in self.buffer:
            sample = entry['sample']
            if sample.sample_id in ids_to_mark and not entry.get('synced', False):
                entry['synced'] = True
                marked += 1
        
        logger.debug(f"Marked {marked} samples as synced in quarantine buffer")
        return marked
        
    def process_validation_results(self, validation_results: List[Dict[str, Any]]) -> int:
        """Process validation results from server.
        
        Args:
            validation_results: List of validation results from server
            
        Returns:
            Number of samples updated with validation results
        """
        # Map of sample IDs to validation results for faster lookups
        validation_map = {result['sample_id']: result for result in validation_results}
        
        # Counter for updated samples
        updated = 0
        
        # Update samples with validation results
        for entry in self.buffer:
            sample = entry['sample']
            if sample.sample_id in validation_map:
                result = validation_map[sample.sample_id]
                
                # Update the sample status
                entry['validated'] = True
                entry['validation_result'] = result
                
                # Update the sample with the validated label if provided
                if 'validated_label' in result:
                    sample.label = result['validated_label']
                    
                updated += 1
        
        logger.debug(f"Updated {updated} samples with validation results in quarantine buffer")
        return updated