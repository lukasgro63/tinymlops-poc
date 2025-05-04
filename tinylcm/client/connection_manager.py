import json
import logging
import time
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests

from tinylcm.utils.errors import ConnectionError
from tinylcm.utils.logging import setup_logger


class ConnectionManager:
    def __init__(self, server_url: str, max_retries: int = 3, retry_delay: float = 1.0, connection_timeout: float = 300.0, use_exponential_backoff: bool = False, backoff_factor: float = 2.0, request_timeout: float = 30.0, headers: Optional[Dict[str, str]] = None):
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.server_url = server_url.rstrip('/')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_timeout = connection_timeout
        self.use_exponential_backoff = use_exponential_backoff
        self.backoff_factor = backoff_factor
        self.request_timeout = request_timeout
        self.headers = headers or {}
        self.connection_status = "disconnected"
        self.last_connection_time = None
        self.failed_attempts = 0
        self.status_callbacks: List[Callable[[str, Optional[str]], None]] = []
        self.logger.info(f"Initialized connection manager for server: {server_url}")
    
    def register_status_callback(self, callback: Callable[[str, Optional[str]], None]) -> None:
        self.status_callbacks.append(callback)
        self.logger.debug(f"Registered status callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def _notify_status_change(self, status: str, info: Optional[str] = None) -> None:
        self.connection_status = status
        for callback in self.status_callbacks:
            try:
                callback(status, info)
            except Exception as e:
                self.logger.error(f"Error in status callback: {str(e)}")
    
    def connect(self) -> bool:
        self.logger.debug(f"Attempting to connect to server: {self.server_url}")
        try:
            response = requests.get(f"{self.server_url}/api/status", headers=self.headers, timeout=self.request_timeout)
            if response.status_code == 200:
                self.logger.info(f"Successfully connected to server: {self.server_url}")
                self.last_connection_time = time.time()
                self.failed_attempts = 0
                self._notify_status_change("connected")
                return True
            else:
                error_msg = f"Server returned unexpected status code: {response.status_code}"
                self.logger.warning(error_msg)
                self.failed_attempts += 1
                self._notify_status_change("error", error_msg)
                return False
        except requests.RequestException as e:
            error_msg = f"Connection error: {str(e)}"
            self.logger.warning(error_msg)
            self.failed_attempts += 1
            self._notify_status_change("error", error_msg)
            return False
    
    def connect_with_retry(self) -> bool:
        if self.connect():
            return True
        retry_count = 1
        while retry_count < self.max_retries:
            if self.use_exponential_backoff:
                delay = self.retry_delay * (self.backoff_factor ** (retry_count - 1))
            else:
                delay = self.retry_delay
            self.logger.info(f"Retrying connection (attempt {retry_count + 1}/{self.max_retries}) after {delay:.2f}s delay")
            self._notify_status_change("retrying", f"Attempt {retry_count + 1}/{self.max_retries}")
            time.sleep(delay)
            if self.connect():
                return True
            retry_count += 1
        error_msg = f"Failed to connect after {self.max_retries} attempts"
        self.logger.error(error_msg)
        raise ConnectionError(error_msg)
    
    def is_connected(self) -> bool:
        if self.last_connection_time is None:
            return False
        if time.time() - self.last_connection_time > self.connection_timeout:
            self.logger.debug("Connection timeout expired, re-checking connection")
            return self.connect()
        return self.connection_status == "connected"
    
    def reset_connection(self) -> None:
        self.logger.info("Resetting connection state")
        self.connection_status = "disconnected"
        self.last_connection_time = None
        self.failed_attempts = 0
        self._notify_status_change("disconnected")
        
    def execute_request(self, method: str, endpoint: str, auto_connect: bool = True, retry_on_failure: bool = True, **kwargs) -> requests.Response:
        if auto_connect and not self.is_connected():
            if retry_on_failure:
                self.connect_with_retry()
            else:
                self.connect()
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]
        
        # Only add api/ if not already present
        if not endpoint.startswith('api/'):
            endpoint = f"api/{endpoint}"
                
        url = f"{self.server_url}/{endpoint}"
        request_headers = self.headers.copy()
        if 'headers' in kwargs:
            request_headers.update(kwargs.pop('headers'))
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.request_timeout
        
        # DEBUGGING: Print request details
        self.logger.debug(f"Making {method} request to {url}")
        if 'files' in kwargs:
            self.logger.debug(f"Request includes files: {list(kwargs['files'].keys())}")
        if 'data' in kwargs:
            self.logger.debug(f"Request includes data parameters: {list(kwargs['data'].keys())}")
        
        try:
            method = method.upper()
            if method == 'GET':
                response = requests.get(url, headers=request_headers, **kwargs)
            elif method == 'POST':
                # IMPORTANT: Get files and data directly from kwargs without popping them
                # This ensures they are passed correctly to requests.post
                response = requests.post(url, headers=request_headers, **kwargs)
            elif method == 'PUT':
                response = requests.put(url, headers=request_headers, **kwargs)
            elif method == 'DELETE':
                response = requests.delete(url, headers=request_headers, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            return response
        except requests.RequestException as e:
            self.logger.error(f"Request error for {method} {url}: {str(e)}")
            if isinstance(e, (requests.ConnectionError, requests.Timeout)):
                self.failed_attempts += 1
                self._notify_status_change("error", str(e))
                if retry_on_failure:
                    self.logger.info("Attempting to reconnect before retrying request")
                    if self.connect_with_retry():
                        self.logger.info("Reconnected successfully, retrying request")
                        return self.execute_request(method=method, endpoint=endpoint, auto_connect=False, retry_on_failure=False, **kwargs)
            raise