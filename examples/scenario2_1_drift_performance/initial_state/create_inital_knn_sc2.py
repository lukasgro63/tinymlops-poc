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

from tinylcm.core.classifiers.knn import LightweightKNN
from tinylcm.utils.file_utils import ensure_directory_exists

# --- KONFIGURATION ---
# Pfade zu initialen Bilddaten
INITIAL_IMAGE_DATA_DIR = root_dir / "examples/assets/training_images"  # Absoluter Pfad zum Training-Images Ordner
# Update auf die neuen Klassen mit Einbeziehung der Ball-Klasse
CLASSES = {
    "negative": INITIAL_IMAGE_DATA_DIR / "negative",
    "stone": INITIAL_IMAGE_DATA_DIR / "stone",
    "lego": INITIAL_IMAGE_DATA_DIR / "lego",
    "leaf": INITIAL_IMAGE_DATA_DIR / "leaf",
    "coin": INITIAL_IMAGE_DATA_DIR / "coin"  # Neue Ball-Klasse
}

# Pfad zum TFLite-Modell (das auch im Beispiel verwendet wird)
# Verwenden wir einen absoluten Pfad, der vom Root-Verzeichnis ausgeht
MODEL_PATH = str(root_dir / "examples/scenario3_drift_objects2/model/model_object.tflite")  # Aktualisiert für Scenario3
LABELS_PATH = str(root_dir / "examples/scenario3_drift_objects2/model/labels_object.txt")  # Aktualisiert für Scenario3

# Pfad zum Feature-Prozessor (StandardScaler + PCA)
FEATURE_PROCESSOR_PATH = str(root_dir / "examples/scenario3_drift_objects2/model/feature_processor.pkl")

# Konfiguration für den Feature Extractor
TARGET_IMG_SIZE = (224, 224)  # Inferenzauflösung

# Konfiguration für den LightweightKNN
KNN_K = 5  # Angepasst für 5 Klassen (4 positive + 1 negative)
KNN_MAX_SAMPLES = 200  # Erhöht für den größeren Datensatz
KNN_DISTANCE_METRIC = "euclidean"  # Metrik für den Abstandsvergleich
KNN_USE_NUMPY = True  # Für die Offline-Erstellung können wir NumPy nutzen

# Speicherort für den initialen k-NN Zustand
OUTPUT_STATE_DIR = current_dir  # Das aktuelle Verzeichnis (initial_state)
OUTPUT_STATE_FILENAME = "knn_initial_state_objects.json"  # Zustandsdatei
OUTPUT_REFERENCE_STATS_FILENAME = "knn_reference_stats.json"  # Datei für die Referenzstatistiken
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

def extract_features_manually(image, interpreter, input_details, output_details, feature_processor_path=None):
    """
    Extrahiere Features manuell mit dem TFLite-Interpreter und wende optional
    den StandardScalerPCA-Prozessor an.
    
    Args:
        image: Das Eingabebild
        interpreter: TFLite-Interpreter
        input_details: Input-Details des Interpreters
        output_details: Output-Details des Interpreters
        feature_processor_path: Pfad zum StandardScaler+PCA-Prozessor
        
    Returns:
        Feature-Vektor (bei Anwendung des Prozessors: reduzierte Dimensionen)
    """
    # Konvertiere das Bild in das richtige Format für das Modell
    processed_image = prepare_input_tensor_quantized(image, input_details)

    # Setze den Input-Tensor und führe die Inferenz durch
    interpreter.set_tensor(input_details['index'], processed_image)
    interpreter.invoke()

    # Hole alle Output-Details und drucke sie aus für Debugzwecke
    all_output_details = interpreter.get_output_details()
    print(f"Modell hat {len(all_output_details)} Output-Tensoren:")
    
    for i, output in enumerate(all_output_details):
        tensor = interpreter.get_tensor(output['index'])
        print(f"  Output {i}: Index={output['index']}, Name={output.get('name', 'unnamed')}, Shape={tensor.shape}")
    
    # Bei unserem MobileNetV2-Modell ist der Output-Tensor der Feature-Vektor
    # Er sollte bereits die korrekte Form haben (1, 1280) nach GlobalAveragePooling2D
    feature_tensor = interpreter.get_tensor(output_details['index'])
    
    # Entferne Batch-Dimension (1, 1280) -> (1280,)
    if len(feature_tensor.shape) > 1 and feature_tensor.shape[0] == 1:
        feature_tensor = feature_tensor[0]
    
    print(f"Extrahierte Features: shape={feature_tensor.shape}, min={feature_tensor.min():.4f}, max={feature_tensor.max():.4f}")
    
    # Wende den Feature-Prozessor an, wenn ein Pfad angegeben wurde
    if feature_processor_path and os.path.exists(feature_processor_path):
        try:
            import pickle

            # Lade den Feature-Prozessor
            print(f"Lade Feature-Prozessor von: {feature_processor_path}")
            with open(feature_processor_path, 'rb') as f:
                processor = pickle.load(f)
            
            # Prüfe, ob ein kompatibles Format vorliegt
            if processor.get('compatible_format', False):
                print(f"Nutze kompatibles Format des Feature-Prozessors (numpy {processor.get('numpy_version', 'unknown')})")
                
                # Rekonstruiere StandardScaler
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler_data = processor['scaler_data']
                scaler.mean_ = np.array(scaler_data['mean'])
                scaler.var_ = np.array(scaler_data['var'])
                scaler.scale_ = np.array(scaler_data['scale'])
                scaler.n_features_in_ = scaler_data['n_features_in']
                scaler.n_samples_seen_ = scaler_data['n_samples_seen']
                
                # Rekonstruiere PCA wenn vorhanden
                pca = None
                if 'pca_data' in processor:
                    from sklearn.decomposition import PCA
                    pca_data = processor['pca_data']
                    pca = PCA()
                    pca.n_components_ = pca_data['n_components']
                    pca.components_ = np.array(pca_data['components'])
                    pca.explained_variance_ = np.array(pca_data['explained_variance'])
                    pca.explained_variance_ratio_ = np.array(pca_data['explained_variance_ratio'])
                    
                    if pca_data['mean'] is not None:
                        pca.mean_ = np.array(pca_data['mean'])
                    
                    if pca_data['singular_values'] is not None:
                        pca.singular_values_ = np.array(pca_data['singular_values'])
                    
                    if pca_data['noise_variance'] is not None:
                        pca.noise_variance_ = pca_data['noise_variance']
                
                # Extrahiere Metadaten
                input_dim = processor.get('input_dim')
                output_dim = processor.get('output_dim')
                
            else:
                # Original Format - direkte Extraktion
                print("Nutze originales Format des Feature-Prozessors")
                scaler = processor.get('scaler')
                pca = processor.get('pca')
                input_dim = processor.get('input_dim')
                output_dim = processor.get('output_dim')
            
            if scaler is None:
                print("WARNUNG: Kein StandardScaler im Prozessor gefunden!")
                return feature_tensor
            
            # Überprüfe die Feature-Dimension
            if len(feature_tensor) != input_dim:
                print(f"WARNUNG: Feature-Dimension stimmt nicht überein: {len(feature_tensor)} vs. erwartet {input_dim}")
                
                # Passe die Feature-Dimension an, wenn nötig
                if len(feature_tensor) > input_dim:
                    print(f"Schneide Features auf {input_dim} Dimensionen")
                    feature_tensor = feature_tensor[:input_dim]
                else:
                    print(f"Fülle Features mit Nullen auf {input_dim} Dimensionen")
                    padded = np.zeros(input_dim)
                    padded[:len(feature_tensor)] = feature_tensor
                    feature_tensor = padded
            
            # Wende StandardScaler an
            scaled_features = scaler.transform(feature_tensor.reshape(1, -1))
            print(f"Features nach StandardScaler: min={scaled_features.min():.4f}, max={scaled_features.max():.4f}")
            
            # Wende PCA an, falls vorhanden
            if pca is not None:
                transformed_features = pca.transform(scaled_features)[0]  # Entferne Batch-Dimension
                print(f"Features nach PCA: shape={transformed_features.shape}, min={transformed_features.min():.4f}, max={transformed_features.max():.4f}")
                return transformed_features
            else:
                # Rückgabe der skalierten Features ohne PCA
                return scaled_features[0]
                
        except Exception as e:
            print(f"Fehler bei der Anwendung des Feature-Prozessors: {e}")
            import traceback
            traceback.print_exc()
            print("Verwende die ursprünglichen Features")
    
    return feature_tensor

def calculate_knn_distance_statistics(knn_instance, all_features, all_labels):
    """Berechne Statistiken über KNN-Distanzen für den KNNDistanceMonitor.
    
    Args:
        knn_instance: Eine trainierte LightweightKNN Instanz
        all_features: Liste von Feature-Vektoren
        all_labels: Liste von Labels für die Feature-Vektoren
        
    Returns:
        Dictionary mit Statistiken (mean, std, etc.)
    """
    print("Berechne KNN-Distanz-Statistiken für initiale Referenz...")
    
    # Stelle sicher, dass KNN bereits Samples enthält
    if len(knn_instance.X_train) == 0:
        print("FEHLER: KNN ist leer, kann keine Distanzen berechnen")
        return {}
    
    # Für jedes Feature in all_features, berechne die Distanzen zu den k nächsten Nachbarn
    all_avg_distances = []
    
    # Gruppiere Features nach Label für bessere Analyse
    avg_distances_by_class = {}
    
    print(f"Berechne Distanzen für {len(all_features)} Samples...")
    
    for i, (feature, label) in enumerate(zip(all_features, all_labels)):
        # Verwende die interne _find_neighbors Methode, um Nachbarn und Distanzen zu finden
        nearest_neighbors = knn_instance._find_neighbors(feature)
        
        # Extrahiere die Distanzen aus den Tupeln (idx, distance)
        distances = [dist for _, dist in nearest_neighbors]
        
        # Berechne durchschnittliche Distanz
        if len(distances) > 0:
            avg_distance = sum(distances) / len(distances)
            all_avg_distances.append(avg_distance)
            
            # Füge zur klassenspezifischen Statistik hinzu
            if label not in avg_distances_by_class:
                avg_distances_by_class[label] = []
            avg_distances_by_class[label].append(avg_distance)
            
            # Log gelegentlich den Fortschritt
            if i % 50 == 0:
                print(f"  Fortschritt: {i}/{len(all_features)} Samples verarbeitet")
                print(f"  Beispiel: Label={label}, Distanzen={distances}, Durchschnitt={avg_distance:.4f}")
    
    # Berechne globale Statistiken
    if len(all_avg_distances) > 0:
        mean_distance = np.mean(all_avg_distances)
        std_distance = np.std(all_avg_distances)
        min_distance = np.min(all_avg_distances)
        max_distance = np.max(all_avg_distances)
        
        print(f"Globale KNN-Distanz-Statistiken:")
        print(f"  Mean: {mean_distance:.4f}")
        print(f"  Std: {std_distance:.4f}")
        print(f"  Min: {min_distance:.4f}")
        print(f"  Max: {max_distance:.4f}")
        
        # Berechne klassenspezifische Statistiken
        class_stats = {}
        for label, distances in avg_distances_by_class.items():
            class_mean = np.mean(distances)
            class_std = np.std(distances)
            
            class_stats[label] = {
                'mean': float(class_mean),
                'std': float(class_std),
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'count': len(distances)
            }
            
            print(f"Klasse {label}: Mean={class_mean:.4f}, Std={class_std:.4f}, Count={len(distances)}")
        
        # Erstelle Statistik-Dictionary
        statistics = {
            'reference_mean': float(mean_distance),
            'reference_std': float(std_distance),
            'min_distance': float(min_distance),
            'max_distance': float(max_distance),
            'sample_count': len(all_avg_distances),
            'class_statistics': class_stats,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        return statistics
    else:
        print("WARNUNG: Keine Distanzen berechnet!")
        return {}

def main():
    print("Erstelle initialen k-NN Zustand für Scenario 3 mit Ball-Klasse...")
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
    if 'quantization' in input_details and len(input_details['quantization']) == 4:
        scale, zero_point = input_details['quantization'][0], input_details['quantization'][1]
        print(f"Input Quantisierung: scale={scale}, zero_point={zero_point}")
    elif 'quantization' in input_details and len(input_details['quantization']) == 2:
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
                # Extrahiere Features und wende den Feature-Prozessor an (StandardScaler + PCA)
                features = extract_features_manually(
                    preprocessed_img, 
                    interpreter, 
                    input_details, 
                    output_details,
                    feature_processor_path=FEATURE_PROCESSOR_PATH
                )
                
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

    # Balanciere die Klassen, damit alle gleichmäßig vertreten sind
    # Gruppiere Samples nach Klassen
    features_by_class = {}
    labels_by_class = {}
    timestamps_by_class = {}

    for i, label in enumerate(all_labels):
        if label not in features_by_class:
            features_by_class[label] = []
            labels_by_class[label] = []
            timestamps_by_class[label] = []

        features_by_class[label].append(initial_features_np[i])
        labels_by_class[label].append(label)
        timestamps_by_class[label].append(all_timestamps[i])

    # Finde die kleinste Klasse
    min_class_size = min(len(samples) for samples in features_by_class.values())
    print(f"Balanciere Klassen auf {min_class_size} Samples pro Klasse")

    # Erstelle balancierte Arrays
    balanced_features = []
    balanced_labels = []
    balanced_timestamps = []

    for label, features in features_by_class.items():
        # Wähle zufällig Samples aus jeder Klasse, maximal min_class_size
        indices = np.random.choice(len(features), min(len(features), min_class_size), replace=False)

        for i in indices:
            balanced_features.append(features_by_class[label][i])
            balanced_labels.append(labels_by_class[label][i])
            balanced_timestamps.append(timestamps_by_class[label][i])

    # Konvertiere zurück zu NumPy-Array
    balanced_features_np = np.array(balanced_features)

    # Statt knn.fit() verwenden wir hier eine kontrollierte Befüllung
    # um sicherzustellen, dass jede Klasse gleichmäßig vertreten ist
    print(f"Befülle KNN kontrolliert mit einer festen Anzahl von Samples pro Klasse")
    
    # Limit pro Klasse = max_samples / Anzahl der Klassen
    samples_per_class = KNN_MAX_SAMPLES // len(CLASSES)
    print(f"Verwende {samples_per_class} Samples pro Klasse für insgesamt {samples_per_class * len(CLASSES)} Samples")
    
    # Reorganisiere balanced_features_np nach Klassen
    features_by_class = {}
    for i, label in enumerate(balanced_labels):
        if label not in features_by_class:
            features_by_class[label] = {
                'features': [],
                'timestamps': []
            }
        features_by_class[label]['features'].append(balanced_features_np[i])
        features_by_class[label]['timestamps'].append(balanced_timestamps[i])
    
    # Füge Samples pro Klasse hinzu
    for label, data in features_by_class.items():
        # Begrenze auf samples_per_class
        count = min(len(data['features']), samples_per_class)
        print(f"Füge {count} Samples für Klasse '{label}' hinzu")
        
        # Wähle die ersten count Samples für jede Klasse
        for i in range(count):
            knn.add_sample(
                feature=data['features'][i],
                label=label,
                timestamp=data['timestamps'][i]
            )
    
    # Hole die Size und Klassen vom KNN
    knn_state = knn.get_state()
    training_size = len(knn_state['X_train'])
    classes = list(set(all_labels))
    
    print(f"Initialer k-NN trainiert mit {training_size} Samples für Klassen: {classes}.")

    # 4. Speichere den Zustand des k-NN
    knn_state_dict = knn.get_state()
    
    # Format für den StateManager
    # Wir müssen sicherstellen, dass wir die tatsächliche Feature-Dimension nach PCA-Transformation verwenden
    # Die samples im KNN (X_train) haben die richtige Dimension
    actual_feature_dimension = len(knn_state_dict['X_train'][0]) if knn_state_dict['X_train'] else feature_dimension
    
    # Prüfen, ob die Dimension der transformierten Features korrekt ist
    print(f"Originale Feature-Dimension vom TFLite-Modell: {feature_dimension}")
    print(f"Tatsächliche Feature-Dimension nach Feature-Prozessor: {actual_feature_dimension}")
    
    state_to_save = {
        "classifier": knn_state_dict,
        "metadata": {
            "description": f"Initial KNN state for Objects Scenario with 5 classes: {list(CLASSES.keys())}",
            "creation_date_iso": datetime.now().isoformat(),
            "feature_dimension": actual_feature_dimension,  # Verwende die tatsächliche Dimension nach Transformation
            "original_dimension": feature_dimension,  # Speichere auch die ursprüngliche Dimension
            "feature_processor": "StandardScaler + PCA(256)" if os.path.exists(FEATURE_PROCESSOR_PATH) else "None",
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
    
    # 5. Berechne und speichere Referenzstatistiken für KNNDistanceMonitor
    print("\nBerechne Referenzstatistiken für KNNDistanceMonitor...")
    
    # Verwende balanced_features_np und balanced_labels für eine ausgewogene Repräsentation
    reference_stats = calculate_knn_distance_statistics(knn, balanced_features_np, balanced_labels)
    
    if reference_stats:
        # Füge Metadaten hinzu
        reference_stats.update({
            "description": f"Reference statistics for KNNDistanceMonitor based on {len(balanced_features_np)} balanced samples",
            "model_path": MODEL_PATH,
            "knn_k": KNN_K,
            "feature_dimension": actual_feature_dimension,
            "classes": list(set(balanced_labels))
        })
        
        # Speichere die Referenzstatistiken
        reference_stats_path = OUTPUT_STATE_DIR / OUTPUT_REFERENCE_STATS_FILENAME
        try:
            with open(reference_stats_path, 'w') as f:
                json.dump(reference_stats, f, indent=4)
            print(f"Referenzstatistiken erfolgreich gespeichert in: {reference_stats_path}")
            
            # Gib Empfehlung für die Konfiguration
            reference_mean = reference_stats['reference_mean']
            # Empfehle einen guten Delta-Wert basierend auf den Klassendistanzen
            suggested_delta = reference_stats['reference_std'] * 0.5
            suggested_lambda = reference_stats['reference_std'] * 5.0
            
            print("\nEmpfohlene KNNDistanceMonitor-Konfiguration basierend auf den Statistiken:")
            print(f'''
    "drift_detectors": [
      {{
        "type": "KNNDistanceMonitor",
        "delta": {suggested_delta:.2f},
        "lambda_threshold": {suggested_lambda:.2f},
        "exit_threshold_factor": 0.7,
        "high_confidence_threshold": 0.9,
        "stable_known_classes": {list(CLASSES.keys())},
        "reference_stats_path": "./initial_state/{OUTPUT_REFERENCE_STATS_FILENAME}",
        "warm_up_samples": 5,
        "reference_update_interval": 100,
        "reference_update_factor": 0.05,
        "pause_reference_update_during_drift": true,
        "drift_cooldown_period": 30,
        "use_adaptive_thresholds": false,
        "allow_runtime_threshold_adaptation": false,
        "adaptive_delta_std_multiplier": 1.5,
        "adaptive_lambda_std_multiplier": 5.0
      }}
    ]
            ''')
            
        except Exception as e:
            print(f"FEHLER beim Speichern der Referenzstatistiken: {e}")
    else:
        print("Keine Referenzstatistiken berechnet oder gespeichert")

if __name__ == "__main__":
    main()