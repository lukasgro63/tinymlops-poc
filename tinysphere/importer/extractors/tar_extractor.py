import tarfile
from pathlib import Path
from typing import List

from tinysphere.importer.extractors.base import PackageExtractor


class TarExtractor(PackageExtractor):
    def can_extract(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.tar'
    
    def extract(self, file_path: Path, target_dir: Path) -> List[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_files = []
        
        with tarfile.open(file_path, 'r') as tar:
            tar.extractall(path=target_dir)
            
            for member in tar.getmembers():
                if member.isfile():
                    extracted_files.append(target_dir / member.name)
        
        return extracted_files