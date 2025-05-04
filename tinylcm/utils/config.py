import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar

from tinylcm.constants import DEFAULT_CONFIG_FILE
from tinylcm.utils.file_utils import load_json, save_json

T = TypeVar('T')

_global_config = None

class ConfigProvider(ABC):
    @abstractmethod
    def get_config_value(self, section: str, key: str, default: Any = None) -> Any:
        pass

    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        pass

class FileConfigProvider(ConfigProvider):
    def __init__(self, config_data: Dict[str, Any]) -> None:
        self._config = config_data

    def get_config_value(self, section: str, key: str, default: Any = None) -> Any:
        if section not in self._config:
            return default
        return self._config[section].get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {}).copy()

class Config:
    def __init__(self) -> None:
        self._config = self._get_default_config()
        self._providers: List[ConfigProvider] = [
            FileConfigProvider(self._config)
        ]

    def _get_default_config(self) -> Dict[str, Any]:
        from tinylcm.constants import DEFAULT_CONFIG
        return DEFAULT_CONFIG.copy()

    def get(self, section: Optional[str] = None, key: Optional[str] = None, default: Optional[T] = None) -> Union[Dict[str, Any], Any, T]:
        if section is None:
            return self._config.copy()
        if section not in self._config:
            return default
        if key is None:
            return self._config[section].copy()
        for provider in self._providers:
            try:
                value = provider.get_config_value(section, key, None)
                if value is not None:
                    return value
            except Exception:
                pass
        return self._config[section].get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        return self.get(component_name, default={})

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        try:
            file_config = load_json(file_path)
            for section, section_values in file_config.items():
                if isinstance(section_values, dict):
                    if section not in self._config:
                        self._config[section] = {}
                    for key, value in section_values.items():
                        self._config[section][key] = value
                else:
                    self._config[section] = section_values
            self._providers.insert(0, FileConfigProvider(file_config))
        except Exception as e:
            logging.warning(f"Failed to load config from {file_path}: {str(e)}")

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        try:
            save_json(self._config, file_path)
        except Exception as e:
            logging.error(f"Failed to save config to {file_path}: {str(e)}")

    @contextmanager
    def component_context(self, component_name: str):
        component_config = self.get_component_config(component_name)
        yield component_config

def get_config() -> Config:
    global _global_config
    if _global_config is None:
        _global_config = Config()
        default_path = os.path.join(os.getcwd(), DEFAULT_CONFIG_FILE)
        if os.path.exists(default_path):
            _global_config.load_from_file(default_path)
    return _global_config

def set_global_config(config: Config) -> None:
    global _global_config
    _global_config = config

def load_config(file_path: Union[str, Path]) -> Config:
    config = Config()
    config.load_from_file(file_path)
    return config
