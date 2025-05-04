from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


class PackageExtractor(ABC):
    @abstractmethod
    def can_extract(self, file_path: Path) -> bool:
        pass
    
    @abstractmethod
    def extract(self, file_path: Path, target_dir: Path) -> List[Path]:
        pass
    
    @staticmethod
    def get_appropriate_extractor(file_path: Path) -> Optional['PackageExtractor']:
        from importer.extractors import (GzipExtractor, TarExtractor,
                                         ZipExtractor)
        
        extractors = [
            TarExtractor(),
            ZipExtractor(),
            GzipExtractor()
        ]
        
        for extractor in extractors:
            if extractor.can_extract(file_path):
                return extractor
        
        return None