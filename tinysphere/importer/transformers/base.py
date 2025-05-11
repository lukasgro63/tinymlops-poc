from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class DataTransformer(ABC):
    @abstractmethod
    def can_transform(self, package_type: str, files: List[Path]) -> bool:
        pass
    
    @abstractmethod
    def transform(self, package_id: str, device_id: str, files: List[Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @staticmethod
    def get_appropriate_transformer(package_type: str, files: List[Path]) -> Optional['DataTransformer']:
        from tinysphere.importer.transformers import (LogsTransformer, MetricsTransformer,
                                           ModelTransformer, DriftTransformer)

        # The order here defines the priority when multiple transformers match
        # especially important for "components" packages that may contain multiple types
        transformers = [
            # Model files have highest priority
            ModelTransformer(),
            # Metrics files have second priority
            MetricsTransformer(),
            # Drift events have third priority
            DriftTransformer(),
            # Log files have lowest priority
            LogsTransformer()
        ]
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Finding appropriate transformer for package type '{package_type}'")
        
        for transformer in transformers:
            if transformer.can_transform(package_type, files):
                logger.info(f"Selected transformer: {transformer.__class__.__name__}")
                return transformer
        
        logger.warning(f"No suitable transformer found for package type '{package_type}'")
        return None
    
    @staticmethod
    def get_all_appropriate_transformers(package_type: str, files: List[Path]) -> List['DataTransformer']:
        """Findet alle Transformer, die das Paket verarbeiten können."""
        from tinysphere.importer.transformers import (LogsTransformer, MetricsTransformer,
                                           ModelTransformer, DriftTransformer)

        transformers = [
            # Model files have highest priority
            ModelTransformer(),
            # Metrics files have second priority
            MetricsTransformer(),
            # Drift events have third priority
            DriftTransformer(),
            # Log files have lowest priority
            LogsTransformer()
        ]
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Finding all appropriate transformers for package type '{package_type}'")
        
        matching_transformers = []
        for transformer in transformers:
            if transformer.can_transform(package_type, files):
                logger.info(f"Found matching transformer: {transformer.__class__.__name__}")
                matching_transformers.append(transformer)
        
        if not matching_transformers:
            logger.warning(f"No suitable transformers found for package type '{package_type}'")
        else:
            logger.info(f"Found {len(matching_transformers)} matching transformers for package type '{package_type}'")
        
        return matching_transformers