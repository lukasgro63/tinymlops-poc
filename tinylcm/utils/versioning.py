import datetime
import hashlib
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Protocol, cast

from tinylcm.utils.file_utils import get_file_size

class VersionInfo(Protocol):
    @property
    def version_id(self) -> str:
        ...
    @property
    def timestamp(self) -> float:
        ...

def generate_timestamp_version(prefix: str = "v_") -> str:
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{timestamp}"

def generate_incremental_version(
    directory: Union[str, Path],
    prefix: str = "v_",
    digits: int = 3
) -> str:
    dir_path = Path(directory)
    if not dir_path.exists():
        os.makedirs(dir_path, exist_ok=True)
        if prefix == "v_":
            return f"{prefix}{1:0{digits}d}"
        else:
            return f"{prefix}1"
    pattern = re.compile(f"^{re.escape(prefix)}(\\d+)$")
    max_version = 0
    for item in os.listdir(dir_path):
        match = pattern.match(item)
        if match:
            try:
                version_num = int(match.group(1))
                max_version = max(max_version, version_num)
            except ValueError:
                continue
    next_version = max_version + 1
    if prefix == "v_":
        return f"{prefix}{next_version:0{digits}d}"
    else:
        return f"{prefix}{next_version}"

def calculate_file_hash(
    file_path: Union[str, Path],
    algorithm: str = "md5",
    buffer_size: int = 65536
) -> str:
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if algorithm == "md5":
        hash_obj = hashlib.md5()
    elif algorithm == "sha1":
        hash_obj = hashlib.sha1()
    elif algorithm == "sha256":
        hash_obj = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    with open(path_obj, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            hash_obj.update(data)
    return hash_obj.hexdigest()

def calculate_content_hash(content: Union[str, bytes], algorithm: str = "md5") -> str:
    if algorithm == "md5":
        hash_obj = hashlib.md5()
    elif algorithm == "sha1":
        hash_obj = hashlib.sha1()
    elif algorithm == "sha256":
        hash_obj = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    if isinstance(content, str):
        content_bytes = content.encode('utf-8')
    else:
        content_bytes = content
    hash_obj.update(content_bytes)
    return hash_obj.hexdigest()

def create_version_info(
    source_file: Optional[Union[str, Path]] = None,
    content: Optional[Union[str, bytes]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    version_id: Optional[str] = None
) -> Dict[str, Any]:
    if source_file is None and content is None:
        raise ValueError("Either source_file or content must be provided")
    version_info = {
        "version_id": version_id or str(uuid.uuid4()),
        "timestamp": time.time(),
        "metadata": metadata or {}
    }
    if source_file is not None:
        path_obj = Path(source_file)
        version_info.update({
            "filename": path_obj.name,
            "file_size_bytes": get_file_size(path_obj),
            "file_hash": calculate_file_hash(path_obj)
        })
    if content is not None:
        content_size = len(content.encode('utf-8') if isinstance(content, str) else content)
        version_info.update({
            "content_hash": calculate_content_hash(content),
            "content_size_bytes": content_size
        })
    return version_info

def compare_versions(version1: Dict[str, Any], version2: Dict[str, Any]) -> bool:
    if "file_hash" in version1 and "file_hash" in version2:
        return version1["file_hash"] == version2["file_hash"]
    if "content_hash" in version1 and "content_hash" in version2:
        return version1["content_hash"] == version2["content_hash"]
    return False

def get_version_diff(
    old_version: Dict[str, Any],
    new_version: Dict[str, Any]
) -> Dict[str, Any]:
    diff = {
        "is_same_content": compare_versions(old_version, new_version),
        "time_difference_seconds": new_version.get("timestamp", 0) - old_version.get("timestamp", 0)
    }
    if "file_size_bytes" in old_version and "file_size_bytes" in new_version:
        diff["size_difference_bytes"] = new_version["file_size_bytes"] - old_version["file_size_bytes"]
    if "content_size_bytes" in old_version and "content_size_bytes" in new_version:
        diff["content_size_difference_bytes"] = (
            new_version["content_size_bytes"] - old_version["content_size_bytes"]
        )
    old_metadata = old_version.get("metadata", {})
    new_metadata = new_version.get("metadata", {})
    metadata_changes = {}
    all_keys = set(old_metadata.keys()) | set(new_metadata.keys())
    for key in all_keys:
        old_value = old_metadata.get(key)
        new_value = new_metadata.get(key)
        if old_value != new_value:
            metadata_changes[key] = {
                "from": old_value,
                "to": new_value
            }
    if metadata_changes:
        diff["metadata_changes"] = metadata_changes
    return diff
