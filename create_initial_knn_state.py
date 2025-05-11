import os
import time
import json
from pathlib import Path
from datetime import datetime
import cv2  # Für das Laden und Vorverarbeiten der Bilder
import numpy as np

# Stelle sicher, dass tinylcm und deine utils importiert werden können
from tinylcm.core.feature_extractors.tflite import TFLiteFeatureExtractor
from tinylcm.core.classifiers.knn import LightweightKNN
from tinylcm.utils.file_utils import ensure_directory_exists

# --- KONFIGURATION ---
# Pfade zu deinen initialen Bilddaten
INITIAL_IMAGE_DATA_DIR = Path("examples/assets/knn/images")  # Hauptordner für initiale Bilder
CLASSES = {"red": INITIAL_IMAGE_DATA_DIR / "red", "green": INITIAL_IMAGE_DATA_DIR / "green"}

# Pfad zum TFLite-Modell (das auch im Beispiel verwendet wird)
MODEL_PATH = "examples/assets/model/model.tflite"  # Relativ zum Skript-Ausführungsort
LABELS_PATH = "examples/assets/model/labels.txt"  # Relativ zum Skript-Ausführungsort

# Konfiguration für den TFLiteFeatureExtractor
FEATURE_LAYER_INDEX = -1  # Oder der Index, den du im Beispiel verwendest
TARGET_IMG_SIZE = (224, 224)  # Inferenzauflösung

# Konfiguration für den LightweightKNN
KNN_K = 3  # Anzahl der nächsten Nachbarn
KNN_MAX_SAMPLES = 20  # Da wir nur 20 Samples haben (10 rot, 10 grün)
KNN_DISTANCE_METRIC = "cosine"  # Metrik für den Abstandsvergleich
KNN_USE_NUMPY = True  # Für die Offline-Erstellung können wir NumPy nutzen

# Speicherort für den initialen k-NN Zustand
OUTPUT_STATE_DIR = Path("examples/assets/initial_states/")
OUTPUT_STATE_FILENAME = "knn_initial_state_RG.json"
# --- ENDE KONFIGURATION ---

def preprocess_image_for_feature_extraction(image_path: Path, target_size: tuple) -> np.ndarray:
    """Lädt, resized und normalisiert ein Bild für den TFLiteFeatureExtractor."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # TFLite erwartet oft RGB
    # Normalisierung für float im Bereich [0,1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    return img_normalized  # Shape: (height, width, channels)

def main():
    print("Erstelle initialen k-NN Zustand...")
    ensure_directory_exists(OUTPUT_STATE_DIR)

    # 1. Initialisiere TFLiteFeatureExtractor
    print(f"Initialisiere TFLiteFeatureExtractor mit Modell: {MODEL_PATH}")
    feature_extractor = TFLiteFeatureExtractor(
        model_path=MODEL_PATH,
        feature_layer_index=FEATURE_LAYER_INDEX
    )
    
    # Feature-Dimension ermitteln (wichtig für k-NN Initialisierung im Beispiel)
    # Erzeuge einen Dummy-Input, um die Dimension zu bekommen
    dummy_input_for_dim_check = np.zeros((1, *TARGET_IMG_SIZE, 3), dtype=np.float32)
    
    extracted_dummy_features = feature_extractor.extract_features(dummy_input_for_dim_check)
    feature_dimension = extracted_dummy_features.shape[-1]  # Letzte Dimension ist Feature-Dim
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
        for image_file in image_folder_path.glob("*.jpg"):  # Oder .png etc.
            try:
                print(f"  Verarbeite: {image_file} für Label: {label_name}")
                # Preprocessing
                preprocessed_img = preprocess_image_for_feature_extraction(image_file, TARGET_IMG_SIZE)
                
                # Der TFLiteFeatureExtractor erwartet oft einen Batch-Input (1, H, W, C)
                preprocessed_img_batch = np.expand_dims(preprocessed_img, axis=0)

                features = feature_extractor.extract_features(preprocessed_img_batch)
                all_features.append(features)
                all_labels.append(label_name)
                all_timestamps.append(time.time() - image_count)  # Simuliere leicht unterschiedliche Zeiten
                image_count += 1
            except Exception as e:
                print(f"    Fehler bei der Verarbeitung von {image_file}: {e}")
    
    if not all_features:
        print("FEHLER: Keine Features extrahiert. Bitte überprüfe Pfade und Bilder.")
        return

    # Konvertiere Feature-Liste in ein NumPy-Array mit passender Form
    # Entferne Batch-Dimension (1, features) -> (features,)
    initial_features_np = np.vstack([f.reshape(-1, feature_dimension) for f in all_features])
    print(f"Form der extrahierten Features: {initial_features_np.shape}")

    # 3. Initialisiere und trainiere den LightweightKNN
    knn = LightweightKNN(
        k=KNN_K, 
        max_samples=KNN_MAX_SAMPLES, 
        distance_metric=KNN_DISTANCE_METRIC,
        use_numpy=KNN_USE_NUMPY
    )
    knn.fit(initial_features_np, all_labels, all_timestamps)
    print(f"Initialer k-NN trainiert mit {knn.get_training_size()} Samples für Klassen: {knn.get_classes()}.")

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