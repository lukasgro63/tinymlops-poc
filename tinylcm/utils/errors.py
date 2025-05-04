class TinyLCMError(Exception):
    pass

class ModelError(TinyLCMError):
    pass

class ModelNotFoundError(ModelError):
    pass

class ModelIntegrityError(ModelError):
    pass

class StorageError(TinyLCMError):
    pass

class StorageAccessError(StorageError):
    pass

class StorageWriteError(StorageError):
    pass

class ConfigError(TinyLCMError):
    pass

class DataLoggerError(TinyLCMError):
    pass

class MonitoringError(TinyLCMError):
    pass

class InvalidInputError(TinyLCMError):
    pass

class SyncError(TinyLCMError):
    pass

class ConnectionError(TinyLCMError):
    pass