import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Füge das Hauptverzeichnis zum Python-Path hinzu
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(root_dir))

import cv2  # Für das Laden und Vorverarbeiten der Bilder
import numpy as np

# Importiere die Hilfsfunktionen aus examples/utils/preprocessors.py
from examples.utils.preprocessors import (prepare_input_tensor_quantized,
                                          resize_image)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
        TFLITE_RUNTIME_AVAILABLE = True
    except ImportError:
        TFLITE_RUNTIME_AVAILABLE = False

# Import später, damit wir die Klassen für den manuellen Fall haben
# Wir werden die Klassen direkt nutzen statt über den TFLiteFeatureExtractor
from tinylcm.core.classifiers.knn import LightweightKNN
from tinylcm.utils.file_utils import ensure_directory_exists

# --- KONFIGURATION ---
# Pfade zu deinen initialen Bilddaten
# Angepasster Pfad für die tatsächliche Verzeichnisstruktur
INITIAL_IMAGE_DATA_DIR = Path("examples/assets/initial_states/images")  # Hauptordner für initiale Bilder
CLASSES = {"red": INITIAL_IMAGE_DATA_DIR / "red", "negative": INITIAL_IMAGE_DATA_DIR / "negative"}

# Pfad zum TFLite-Modell (das auch im Beispiel verwendet wird)
MODEL_PATH = "examples/assets/model/model.tflite"  # Relativ zum Skript-Ausführungsort
LABELS_PATH = "examples/assets/model/labels.txt"  # Relativ zum Skript-Ausführungsort

# Konfiguration für den Feature Extractor
FEATURE_LAYER_INDEX = -1  # Oder der Index, den du im Beispiel verwendest
TARGET_IMG_SIZE = (224, 224)  # Inferenzauflösung

# Konfiguration für den LightweightKNN
KNN_K = 3  # Anzahl der nächsten Nachbarn
KNN_MAX_SAMPLES = 100  # Da wir nur 20 Samples haben (10 rot, 10 grün)
KNN_DISTANCE_METRIC = "cosine"  # Metrik für den Abstandsvergleich
KNN_USE_NUMPY = True  # Für die Offline-Erstellung können wir NumPy nutzen

# Speicherort für den initialen k-NN Zustand
OUTPUT_STATE_DIR = Path("examples/assets/initial_states/")
OUTPUT_STATE_FILENAME = "knn_initial_state_RN.json"  # Red, Negative
# --- ENDE KONFIGURATION ---

def preprocess_image_for_feature_extraction(image_path: Path, target_size: tuple) -> np.ndarray:
    """Lädt, resized und normalisiert ein Bild für die Feature-Extraktion."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    
    # Verwende die resize_image Funktion aus utils.preprocessors
    img_resized = resize_image(img, target_size)
    
    # Konvertiere zu RGB (TFLite erwartet oft RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    return img_rgb  # Return RGB image as uint8

def extract_features_manually(image, interpreter, input_details, output_details):
    """Extrahiere Features manuell mit dem TFLite-Interpreter."""
    # Konvertiere das Bild in das richtige Format für das Modell
    # Verwende die prepare_input_tensor_quantized Funktion, die das richtige Datentyp-Handling übernimmt
    processed_image = prepare_input_tensor_quantized(image, input_details)
    
    # Setze den Input-Tensor - input_details ist bereits ein Dictionary, nicht ein Array von Dictionaries
    interpreter.set_tensor(input_details['index'], processed_image)
    
    # Führe die Inferenz durch
    interpreter.invoke()
    
    # Feature-Extraktion (letzter Layer)
    output_tensor = interpreter.get_tensor(output_details['index'])
    
    return output_tensor

def main():
    print("Erstelle initialen k-NN Zustand...")
    ensure_directory_exists(OUTPUT_STATE_DIR)

    # 1. Initialisiere TFLite Interpreter
    print(f"Initialisiere TFLite Interpreter mit Modell: {MODEL_PATH}")
    
    # Lade das Model mit dem korrekten Interpreter
    if TF_AVAILABLE:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    elif TFLITE_RUNTIME_AVAILABLE:
        interpreter = TFLiteInterpreter(model_path=MODEL_PATH)
    else:
        raise ImportError("Weder TensorFlow noch tflite_runtime ist verfügbar. Bitte installiere eines davon.")
    
    interpreter.allocate_tensors()
    
    # Hole Input- und Output-Details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"Input Shape: {input_details['shape']}")
    print(f"Output Shape: {output_details['shape']}")
    print(f"Input Dtype: {input_details['dtype']}")
    
    # WICHTIG: Beachte die Quantisierungsparameter, wenn vorhanden
    if 'quantization' in input_details and len(input_details['quantization']) == 2:
        scale, zero_point = input_details['quantization']
        print(f"Input Quantisierung: scale={scale}, zero_point={zero_point}")

    # Erstelle einen Dummy-Input des richtigen Typs
    dummy_shape = input_details['shape']
    dummy_dtype = input_details['dtype']
    
    # Für UINT8 (quantisiertes Modell) - erzeugen wir ein leeres uint8-Bild
    dummy_input = np.zeros(dummy_shape, dtype=dummy_dtype)
    
    # Wir verwenden prepare_input_tensor_quantized, um die richtige Vorverarbeitung zu erhalten
    dummy_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    processed_dummy = prepare_input_tensor_quantized(dummy_rgb, input_details)
    
    # Setze den Input-Tensor mit dem korrekt verarbeiteten Dummy-Input
    interpreter.set_tensor(input_details['index'], processed_dummy)
    
    # Führe die Inferenz durch
    interpreter.invoke()
    
    # Extrahiere Features vom Dummy-Input
    dummy_output = interpreter.get_tensor(output_details['index'])
    
    # Bestimme die Feature-Dimension (entferne Batch-Dimension wenn vorhanden)
    if len(dummy_output.shape) > 1:
        feature_dimension = dummy_output.shape[1]
    else:
        feature_dimension = dummy_output.shape[0]
        
    print(f"Ermittelte Feature-Dimension: {feature_dimension}")

    # 2. Extrahiere Features für deine initialen Bilder
    all_features = []
    all_labels = []
    all_timestamps = []

    print("Extrahiere Features aus initialen Bildern...")
    for label_name, image_folder_path in CLASSES.items():
        if not image_folder_path.is_dir():
            print(f"WARNUNG: Ordner für Klasse '{label_name}' nicht gefunden: {image_folder_path}")
            continue
        
        image_count = 0
        # Unterstütze mehrere Bildformate
        image_files = list(image_folder_path.glob("*.jpg")) + list(image_folder_path.glob("*.png"))
        
        if not image_files:
            print(f"WARNUNG: Keine Bilder im Ordner '{label_name}' gefunden: {image_folder_path}")
            continue
            
        for image_file in image_files:
            try:
                print(f"  Verarbeite: {image_file} für Label: {label_name}")
                
                # Preprocessing - Lade und resize das Bild
                preprocessed_img = preprocess_image_for_feature_extraction(image_file, TARGET_IMG_SIZE)
                
                # Feature-Extraktion mit preprocessors.prepare_input_tensor_quantized für korrektes Handling
                features = extract_features_manually(preprocessed_img, interpreter, input_details, output_details)
                
                # Füge Daten hinzu
                all_features.append(features.flatten())  # Flatten, um konsistente 1D-Features zu haben
                all_labels.append(label_name)
                all_timestamps.append(time.time() - image_count)
                image_count += 1
            except Exception as e:
                print(f"    Fehler bei der Verarbeitung von {image_file}: {e}")
                import traceback
                traceback.print_exc()
    
    if not all_features:
        print("FEHLER: Keine Features extrahiert. Bitte überprüfe Pfade und Bilder.")
        return

    # Konvertiere Feature-Liste in ein NumPy-Array
    initial_features_np = np.vstack(all_features)
    print(f"Form der verarbeiteten Features: {initial_features_np.shape}")

    # 3. Initialisiere und trainiere den LightweightKNN
    knn = LightweightKNN(
        k=KNN_K, 
        max_samples=KNN_MAX_SAMPLES, 
        distance_metric=KNN_DISTANCE_METRIC,
        use_numpy=KNN_USE_NUMPY
    )
    
    # Trainiere den KNN mit den extrahierten Features
    knn.fit(initial_features_np, all_labels, all_timestamps)
    
    # Hole die Size und Klassen vom KNN
    # Verwende die korrekten Methoden für LightweightKNN
    knn_state = knn.get_state()
    training_size = len(knn_state['X_train'])  # Die korrigierte Zeile - X_train statt samples
    classes = list(set(all_labels))
    
    print(f"Initialer k-NN trainiert mit {training_size} Samples für Klassen: {classes}.")

    # 4. Speichere den Zustand des k-NN
    knn_state_dict = knn.get_state()
    
    # Format für den StateManager
    state_to_save = {
        "classifier": knn_state_dict,
        "metadata": {
            "description": f"Initial KNN state for classes: {list(CLASSES.keys())}",
            "creation_date_iso": datetime.now().isoformat(),
            "feature_dimension": feature_dimension,
            "source_model": MODEL_PATH
        }
    }
    
    output_file_path = OUTPUT_STATE_DIR / OUTPUT_STATE_FILENAME
    try:
        with open(output_file_path, 'w') as f:
            json.dump(state_to_save, f, indent=4)
        print(f"Initialer k-NN Zustand erfolgreich gespeichert in: {output_file_path}")
    except Exception as e:
        print(f"FEHLER beim Speichern des k-NN Zustands: {e}")

if __name__ == "__main__":
    main()