#!/usr/bin/env python3
"""
TFLite Converter für vortrainiertes MobileNetV2-Modell mit int8-Quantisierung
"""

import os

import numpy as np
import tensorflow as tf

print(f"Using TensorFlow version: {tf.__version__}")

# Modellpfade
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_H5_PATH = os.path.join(MODEL_DIR, "model_object_features.h5")
MODEL_TFLITE_PATH = os.path.join(MODEL_DIR, "model_object.tflite")

# Prüfe, ob das Modell existiert
if not os.path.exists(MODEL_H5_PATH):
    raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_H5_PATH}")

print(f"Lade Modell von: {MODEL_H5_PATH}")
model = tf.keras.models.load_model(MODEL_H5_PATH, compile=False)
print("Modell erfolgreich geladen")

# Zeige Modellzusammenfassung
model.summary()

# Bildgröße für MobileNetV2
IMAGE_SIZE = (224, 224, 3)

# Definiere eine sehr einfache representative dataset Funktion
# mit zufälligen Daten statt echter Validierungsdaten
def representative_data_gen():
    num_calibration_examples = 50  # reduziert von 100 für schnellere Konvertierung
    print(f"Generiere {num_calibration_examples} Kalibrierungsbeispiele...")
    
    for i in range(num_calibration_examples):
        # Erzeuge zufällige Bilddaten im Bereich [0, 255]
        random_input = np.random.randint(0, 256, IMAGE_SIZE, dtype=np.uint8)
        # Erweitere die Dimensionen für den Batch
        random_input = np.expand_dims(random_input, axis=0)
        # Normalisiere auf [0, 1]
        normalized_input = random_input / 255.0
        
        yield [normalized_input]
        
        if (i + 1) % 10 == 0:
            print(f"  Verarbeitet {i + 1}/{num_calibration_examples} Beispiele")

print("\n==== Konvertiere MobileNetV2-Modell zu TFLite mit int8-Quantisierung ====")

# TFLite-Konverter initialisieren
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Aktiviere int8-Quantisierung
print("Int8-Quantisierung wird angewendet, um Modellgröße zu reduzieren...")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Konfiguriere int8 Operationen
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,  # Int8-Operationen für CPU
    tf.lite.OpsSet.TFLITE_BUILTINS         # Reguläre Operationen als Fallback
]

# Setze Inferenz-Eingabe/Ausgabe-Typ
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32  # Features als float32 behalten

# Setze repräsentative Daten für Kalibrierung
converter.representative_dataset = representative_data_gen

try:
    print("Konvertiere Modell zu TFLite-Format...")
    tflite_model = converter.convert()
    
    # Speichere das TFLite-Modell
    with open(MODEL_TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    model_size_mb = os.path.getsize(MODEL_TFLITE_PATH) / (1024 * 1024)
    print(f"\n✅ TFLite-Modell gespeichert unter: {MODEL_TFLITE_PATH}")
    print(f"   Modellgröße: {model_size_mb:.2f} MB")
    
    print("\nQuantisiertes Modell ist optimiert für Raspberry Pi Zero und bietet:")
    print("1. Höhere Feature-Dimensionalität (1280D) - exzellent für Drift-Erkennung")
    print("2. Reduzierte Modellgröße durch int8-Quantisierung")
    print("3. Schnellere Inferenz auf der CPU")
    
except Exception as e:
    print(f"\n❌ FEHLER bei der Konvertierung: {e}")
    print("\nFallback-Optionen:")
    print("1. Versuchen Sie, die Anzahl der Kalibrierungsbeispiele zu reduzieren")
    print("2. Versuchen Sie die Konvertierung ohne Quantisierung mit:")
    print("   converter.optimizations = []")
    print("   converter.representative_dataset = None")