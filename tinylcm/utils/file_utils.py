import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar, Protocol, overload
from typing import Generator, Iterable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
T = TypeVar('T')

def ensure_dir(dir_path: PathLike) -> Path:
    path_obj = Path(dir_path)
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path_obj}")
    except OSError as e:
        logger.error(f"Failed to create or access directory {path_obj}: {e}")
        raise
    return path_obj

def get_file_size(file_path: PathLike) -> int:
    path_obj = Path(file_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"File not found or not a regular file: {file_path}")
    try:
        size = path_obj.stat().st_size
        logger.debug(f"Size of file {path_obj}: {size} bytes")
        return size
    except OSError as e:
        logger.error(f"Could not get size for file {path_obj}: {e}")
        raise FileNotFoundError(f"Could not access file stats for: {file_path}") from e

@overload
def load_json(file_path: PathLike) -> Dict[str, Any]: ...
@overload
def load_json(file_path: PathLike, default: T) -> Union[Dict[str, Any], T]: ...

def load_json(file_path: PathLike, default: Optional[T] = None) -> Union[Dict[str, Any], T]:
    path_obj = Path(file_path)
    if not path_obj.is_file():
        if default is not None:
            logger.warning(f"JSON file not found at {path_obj}, returning default.")
            return default
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    try:
        with path_obj.open('r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                if default is not None:
                    logger.warning(f"JSON file is empty at {path_obj}, returning default.")
                    return default
                raise ValueError(f"JSON file is empty: {file_path}")
            data = json.loads(content)
            logger.debug(f"Successfully loaded JSON from: {path_obj}")
            return data
    except json.JSONDecodeError as e:
        if default is not None:
            logger.warning(f"Invalid JSON in file {path_obj} (Error: {e}), returning default.")
            return default
        logger.error(f"Failed to decode JSON from {path_obj}: {e}")
        raise
    except OSError as e:
        if default is not None:
            logger.warning(f"Could not read file {path_obj} (Error: {e}), returning default.")
            return default
        logger.error(f"Error reading file {path_obj}: {e}")
        raise FileNotFoundError(f"Could not read file: {file_path}") from e

def save_json(data: Dict[str, Any], file_path: PathLike, pretty: bool = True) -> None:
    path_obj = Path(file_path)
    ensure_dir(path_obj.parent)
    try:
        with path_obj.open('w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        logger.debug(f"Successfully saved JSON to: {path_obj}")
    except TypeError as e:
        logger.error(f"Data for {path_obj} is not JSON serializable: {e}", exc_info=True)
        raise
    except OSError as e:
        logger.error(f"Failed to write JSON to {path_obj}: {e}")
        raise

def list_files(
    directory: PathLike,
    pattern: str = "*",
    recursive: bool = False,
    absolute: bool = False
) -> List[Path]:
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.warning(f"Directory not found for listing files: {dir_path}")
        return []
    glob_pattern = f"**/{pattern}" if recursive else pattern
    logger.debug(f"Listing files in '{dir_path}' matching '{glob_pattern}' (Recursive={recursive})")
    try:
        matched_items = list(dir_path.glob(glob_pattern))
        matched_files = [item for item in matched_items if item.is_file()]
    except Exception as e:
        logger.error(f"Error listing files in {dir_path} with pattern {glob_pattern}: {e}", exc_info=True)
        return []
    if absolute:
        result_paths = [f.resolve() for f in matched_files]
    else:
        result_paths = matched_files
    logger.debug(f"Found {len(result_paths)} matching files.")
    return result_paths

def safe_remove(path: PathLike) -> bool:
    path_obj = Path(path)
    try:
        if path_obj.is_symlink():
            logger.debug(f"Removing symbolic link: {path_obj}")
            path_obj.unlink()
        elif path_obj.is_file():
            logger.debug(f"Removing file: {path_obj}")
            path_obj.unlink()
        elif path_obj.is_dir():
            logger.debug(f"Removing directory tree: {path_obj}")
            shutil.rmtree(path_obj)
        else:
            logger.debug(f"Item to remove does not exist: {path_obj}")
            return True
        logger.info(f"Successfully removed: {path_obj}")
        return True
    except OSError as e:
        logger.warning(f"Could not remove {path_obj}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error removing {path_obj}: {e}", exc_info=True)
        return False

def stream_read(
    file_path: PathLike,
    chunk_size: int = 1024,
    mode: str = 'r'
) -> Generator[str, None, None]:
    path_obj = Path(file_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    encoding = 'utf-8' if 'b' not in mode else None
    try:
        with path_obj.open(mode=mode, encoding=encoding) as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        logger.error(f"Error reading file {path_obj}: {e}")
        raise

def stream_write(
    data_generator: Iterable[Any],
    file_path: PathLike,
    mode: str = 'w'
) -> None:
    path_obj = Path(file_path)
    ensure_dir(path_obj.parent)
    encoding = 'utf-8' if 'b' not in mode else None
    try:
        with path_obj.open(mode=mode, encoding=encoding) as f:
            for chunk in data_generator:
                f.write(chunk)
    except Exception as e:
        logger.error(f"Error writing to file {path_obj}: {e}")
        from tinylcm.utils.errors import StorageWriteError
        raise StorageWriteError(f"Failed to write to {file_path}: {e}")

def stream_read_jsonl(file_path: PathLike) -> Generator[Dict[str, Any], None, None]:
    path_obj = Path(file_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    try:
        with path_obj.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {path_obj}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading JSONL file {path_obj}: {e}")
        raise

def stream_write_jsonl(
    data_generator: Iterable[Dict[str, Any]],
    file_path: PathLike
) -> None:
    path_obj = Path(file_path)
    ensure_dir(path_obj.parent)
    try:
        with path_obj.open('w', encoding='utf-8') as f:
            for item in data_generator:
                f.write(json.dumps(item))
                f.write('\n')
    except Exception as e:
        logger.error(f"Error writing JSONL to {path_obj}: {e}")
        from tinylcm.utils.errors import StorageWriteError
        raise StorageWriteError(f"Failed to write JSONL to {file_path}: {e}")