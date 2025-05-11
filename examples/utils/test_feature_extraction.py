#!/usr/bin/env python3
"""
Test-Script für die Feature-Extraktion aus einem TFLite-Modell
-------------------------------------------------------------
Dieses Script testet verschiedene Layer-Indizes für die Feature-Extraktion
und analysiert die resultierenden Feature-Vektoren.
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Pfad zum übergeordneten Verzeichnis hinzufügen
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import der TinyLCM-Komponenten
from tinylcm.core.feature_extractors.tflite import TFLiteFeatureExtractor
from utils.preprocessors import resize_image

def preprocess_image(image):
    """Vorverarbeitung des Bildes für das TFLite-Modell."""
    # Umwandlung in RGB, falls notwendig
    if image.ndim == 3 and image.shape[2] == 3:
        # Prüfen, ob das Bild bereits RGB ist (nicht BGR)
        if isinstance(image, np.ndarray) and image.flags['C_CONTIGUOUS']:
            # Bild ist wahrscheinlich BGR von OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        # BGRA zu RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    return image

def load_image(image_path, target_size=(224, 224)):
    """Laden und Vorbereiten eines Bildes für die Inferenz."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Konnte Bild nicht laden: {image_path}")
    
    # Bild auf Zielgröße resizen
    resized_img = resize_image(img, target_size)
    
    return resized_img

def analyze_features(features, layer_idx, save_path=None):
    """Analysiert und visualisiert Feature-Vektoren."""
    print(f"\nFeature-Analyse für Layer-Index {layer_idx}:")
    print(f"  Shape: {features.shape}")
    print(f"  Datentyp: {features.dtype}")
    print(f"  Min: {np.min(features)}")
    print(f"  Max: {np.max(features)}")
    print(f"  Mean: {np.mean(features)}")
    print(f"  Std: {np.std(features)}")
    
    # Berechne Histogramm der Feature-Werte
    plt.figure(figsize=(10, 6))
    plt.hist(features.flatten(), bins=50)
    plt.title(f'Feature-Verteilung für Layer-Index {layer_idx}')
    plt.xlabel('Feature-Wert')
    plt.ylabel('Häufigkeit')
    
    if save_path:
        # Erstelle Verzeichnis, falls es nicht existiert
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"  Histogramm gespeichert unter: {save_path}")
    
    plt.close()
    
    # Ausgabe der ersten paar Feature-Werte
    print(f"  Erste 10 Feature-Werte: {features.flatten()[:10]}")
    
    return {
        "shape": features.shape,
        "min": float(np.min(features)),
        "max": float(np.max(features)),
        "mean": float(np.mean(features)),
        "std": float(np.std(features)),
        "first_10": features.flatten()[:10].tolist()
    }

def test_layer_indices(model_path, image_paths, layer_indices=None, normalize=False, output_dir="./feature_analysis"):
    """Testet verschiedene Layer-Indizes für die Feature-Extraktion."""
    if layer_indices is None:
        # Standard-Layer-Indizes testen
        layer_indices = [-1, -2, -3, -4, -5, 177]  # 177 ist die vorletzte Layer (basierend auf der Modellanalyse)
    
    print(f"Teste Feature-Extraktion mit Modell: {model_path}")
    print(f"Bilder: {len(image_paths)}")
    print(f"Layer-Indizes: {layer_indices}")
    print(f"Normalisierung: {'Ja' if normalize else 'Nein'}")
    
    # Ausgabeverzeichnis erstellen
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    results = {}
    
    # Test für jeden Layer-Index
    for layer_idx in layer_indices:
        try:
            print(f"\nTesten von Layer-Index: {layer_idx}")
            
            # Feature-Extraktor mit aktuellem Layer-Index initialisieren
            extractor = TFLiteFeatureExtractor(
                model_path=model_path,
                feature_layer_index=layer_idx,
                normalize_features=normalize,
                preprocessors=[preprocess_image]
            )
            
            layer_results = []
            
            # Feature-Extraktion für jedes Bild
            for i, img_path in enumerate(image_paths):
                try:
                    print(f"  Verarbeite Bild {i+1}/{len(image_paths)}: {img_path}")
                    
                    # Bild laden und vorbereiten
                    img = load_image(img_path)
                    
                    # Features extrahieren
                    start_time = time.time()
                    features = extractor.extract_features(img)
                    extraction_time = time.time() - start_time
                    
                    print(f"  Feature-Extraktion in {extraction_time:.4f} Sekunden")
                    
                    # Feature-Analyse
                    img_name = os.path.basename(img_path).split('.')[0]
                    hist_path = os.path.join(output_dir, f"layer_{layer_idx}_{img_name}_hist.png")
                    feature_analysis = analyze_features(features, layer_idx, hist_path)
                    
                    # Speichere Feature-Vektor als NPY-Datei für spätere Analysen
                    feature_path = os.path.join(output_dir, f"layer_{layer_idx}_{img_name}_features.npy")
                    np.save(feature_path, features)
                    print(f"  Feature-Vektor gespeichert unter: {feature_path}")
                    
                    feature_analysis["extraction_time"] = extraction_time
                    layer_results.append(feature_analysis)
                    
                except Exception as e:
                    print(f"  Fehler bei der Verarbeitung von Bild {img_path}: {e}")
            
            results[f"layer_{layer_idx}"] = layer_results
            
        except Exception as e:
            print(f"Fehler bei Layer-Index {layer_idx}: {e}")
    
    # Speichere Gesamtergebnisse als Zusammenfassung
    summary_path = os.path.join(output_dir, "feature_extraction_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Feature-Extraktion Zusammenfassung\n")
        f.write("===============================\n\n")
        
        for layer_key, layer_results in results.items():
            f.write(f"Layer: {layer_key}\n")
            f.write("-" * 40 + "\n")
            
            for i, result in enumerate(layer_results):
                img_path = image_paths[i] if i < len(image_paths) else "Unbekannt"
                img_name = os.path.basename(img_path).split('.')[0]
                
                f.write(f"Bild: {img_name}\n")
                f.write(f"  Shape: {result['shape']}\n")
                f.write(f"  Min: {result['min']}\n")
                f.write(f"  Max: {result['max']}\n")
                f.write(f"  Mean: {result['mean']}\n")
                f.write(f"  Std: {result['std']}\n")
                f.write(f"  Extraktion-Zeit: {result['extraction_time']:.4f} Sekunden\n")
                f.write("\n")
            
            f.write("\n\n")
    
    print(f"Zusammenfassung gespeichert unter: {summary_path}")
    return results

def main():
    """Hauptfunktion zum Testen der Feature-Extraktion."""
    parser = argparse.ArgumentParser(description="Test-Tool für TFLite Feature-Extraktion")
    parser.add_argument("--model", type=str, default="../assets/model/model.tflite",
                       help="Pfad zum TFLite-Modell")
    parser.add_argument("--images", type=str, nargs='+', required=True,
                       help="Pfade zu Testbildern (kann mehrere sein)")
    parser.add_argument("--layers", type=int, nargs='+', default=[-1, -2, -3, -4, 177],
                       help="Layer-Indizes zum Testen")
    parser.add_argument("--normalize", action="store_true",
                       help="Features normalisieren")
    parser.add_argument("--output", type=str, default="./feature_analysis",
                       help="Ausgabeverzeichnis für die Analyseergebnisse")
    
    args = parser.parse_args()
    
    # Prüfen, ob die Bilder existieren
    valid_images = []
    for img_path in args.images:
        if os.path.exists(img_path):
            valid_images.append(img_path)
        else:
            print(f"Warnung: Bild nicht gefunden: {img_path}")
    
    if not valid_images:
        print("Fehler: Keine gültigen Bilder zum Testen gefunden!")
        return
    
    # Feature-Extraktion testen
    test_layer_indices(
        model_path=args.model,
        image_paths=valid_images,
        layer_indices=args.layers,
        normalize=args.normalize,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()