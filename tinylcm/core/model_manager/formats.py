from enum import Enum

class ModelFormat(str, Enum):
    TFLITE = "tflite"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    PICKLE = "pkl"
    JSON = "json"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, format_str: str) -> "ModelFormat":
        try:
            return cls(format_str.lower())
        except ValueError:
            return cls.UNKNOWN