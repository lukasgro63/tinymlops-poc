# Speichern als diagnose.py
import sys

print(f"Python-Version: {sys.version}")
print(f"Pythonpfad: {sys.path}")

try:
    import tensorflow
    print(f"TensorFlow-Pfad: {tensorflow.__file__}")
    print(f"TensorFlow-Attribute: {dir(tensorflow)}")
except Exception as e:
    print(f"TensorFlow-Importfehler: {e}")