"""
Feature Processor für TinyLCM

Diese Klasse implementiert einen Feature-Prozessor für TinyLCM, der StandardScaler und PCA
zur Dimensionsreduktion kombiniert. Er wird nach der Feature-Extraktion und vor der
KNN-Klassifikation eingesetzt, um:

1. Die Feature-Vektoren zu standardisieren (Mittelwert 0, Std 1)
2. Die Dimensionalität mittels PCA zu reduzieren (z.B. von 1280D auf 256D)

Dies verbessert sowohl die Performance (schnellere KNN-Berechnungen) als auch die
Drift-Erkennung auf ressourcenbeschränkten Geräten wie dem Raspberry Pi Zero.
"""

import os
import pickle
import logging
import numpy as np

class FeatureProcessor:
    """Verarbeitet Features durch Standardisierung und PCA-Dimensionsreduktion."""
    
    def __init__(self, processor_path, enable_logging=True):
        """
        Initialisiert den Feature-Prozessor mit einem gespeicherten Prozessor-Modell.
        
        Args:
            processor_path: Pfad zur gespeicherten Prozessor-Datei (.pkl)
            enable_logging: Ob Logging aktiviert werden soll (Default: True)
        """
        self.enabled = False
        self.logger = logging.getLogger("FeatureProcessor") if enable_logging else None
        
        try:
            self._log("Loading feature processor from: " + processor_path)
            
            with open(processor_path, 'rb') as f:
                self.processor = pickle.load(f)
            
            # Extrahiere Komponenten und Metadaten
            self.scaler = self.processor.get('scaler')
            self.pca = self.processor.get('pca')
            self.input_dim = self.processor.get('input_dim')
            self.output_dim = self.processor.get('output_dim')
            self.explained_variance = self.processor.get('explained_variance', 0)
            
            # Überprüfe, ob alle benötigten Komponenten vorhanden sind
            if self.scaler is None:
                self._log("WARNING: No scaler found in processor", level="warning")
                return
                
            # PCA ist optional - wenn nicht vorhanden, machen wir nur Standardisierung
            self.pca_enabled = self.pca is not None
            
            # Aktiviere den Prozessor, wenn mindestens der Scaler vorhanden ist
            self.enabled = True
            
            # Log-Info
            if self.pca_enabled:
                self._log(f"Feature processor initialized: {self.input_dim}D → {self.output_dim}D")
                self._log(f"PCA explains {self.explained_variance:.2f}% of variance")
            else:
                self._log("Feature processor initialized with standardization only (no PCA)")
                
        except Exception as e:
            self._log(f"ERROR initializing feature processor: {e}", level="error")
    
    def _log(self, message, level="info"):
        """Hilfsfunktion für Logging mit verschiedenen Levels."""
        if not self.logger:
            return
            
        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
    
    def process(self, features):
        """
        Transformiert Features mittels Standardisierung und optionaler PCA.
        
        Args:
            features: Feature-Vektor oder Array
            
        Returns:
            Verarbeitete Features (standardisiert und ggf. dimensionsreduziert)
        """
        if not self.enabled:
            return features
        
        try:
            # Sicherstellen, dass Features flach sind
            original_shape = features.shape if hasattr(features, 'shape') else None
            flat_features = features.flatten() if hasattr(features, 'flatten') else features
            
            # Dimensionscheck
            if len(flat_features) != self.input_dim:
                self._log(f"Feature dimension mismatch: got {len(flat_features)}, expected {self.input_dim}", 
                         level="warning")
                
                # Versuche Dimensionen anzupassen
                if len(flat_features) > self.input_dim:
                    self._log(f"Truncating features to match expected dimensions")
                    flat_features = flat_features[:self.input_dim]
                else:
                    self._log(f"Padding features with zeros to match expected dimensions")
                    padded = np.zeros(self.input_dim)
                    padded[:len(flat_features)] = flat_features
                    flat_features = padded
            
            # Reshape für Transformation
            features_reshaped = flat_features.reshape(1, -1)
            
            # Standardisierung anwenden
            scaled_features = self.scaler.transform(features_reshaped)
            
            # PCA anwenden, falls vorhanden
            if self.pca_enabled:
                reduced_features = self.pca.transform(scaled_features)[0]
                self._log(f"Features processed: {original_shape or len(flat_features)} → {reduced_features.shape}", 
                         level="debug")
                return reduced_features
            else:
                # Nur standardisierte Features zurückgeben
                self._log(f"Features standardized (no PCA)", level="debug")
                return scaled_features[0]
                
        except Exception as e:
            self._log(f"ERROR processing features: {e}", level="error")
            # Im Fehlerfall Original-Features zurückgeben
            return features