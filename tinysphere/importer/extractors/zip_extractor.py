import zipfile
from pathlib import Path
from typing import List

from tinysphere.importer.extractors.base import PackageExtractor


class ZipExtractor(PackageExtractor):
    def can_extract(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.zip'
    
    def extract(self, file_path: Path, target_dir: Path) -> List[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_files = []
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
            for zip_info in zip_ref.infolist():
                if not zip_info.is_dir():
                    extracted_files.append(target_dir / zip_info.filename)
        
        return extracted_files