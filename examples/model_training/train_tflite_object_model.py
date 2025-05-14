#!/usr/bin/env python3
"""
TensorFlow Transfer Learning for TinyLCM Object Classification (MobileNetV2)

This script trains a transfer learning model for object classification using
images from examples/assets/training_images. It creates a model that:
1. Uses MobileNetV2 as the base model (very efficient for Raspberry Pi Zero)
2. Adds custom classification layers
3. Extracts intermediate features with high dimensionality (ideal for drift detection)
4. Exports to TFLite format with quantization
5. Is compatible with the TinyLCM pipeline

The model is saved in a format compatible with scenario2_drift_objects.
"""

import os
import pathlib
import subprocess
import sys

import numpy as np

# Check for required dependencies and install if missing
required_packages = ["Pillow", "scipy", "scikit-learn", "pandas"]
missing_packages = []

# Check for PIL/Pillow
try:
    from PIL import Image
    print("✓ Pillow is already installed.")
except ImportError:
    missing_packages.append("Pillow")

# Check for SciPy
try:
    import scipy
    print("✓ SciPy is already installed.")
except ImportError:
    missing_packages.append("scipy")

# Check for scikit-learn
try:
    import sklearn
    print("✓ scikit-learn is already installed.")
except ImportError:
    missing_packages.append("scikit-learn")
    
# Check for pandas
try:
    import pandas as pd
    print("✓ pandas is already installed.")
except ImportError:
    print("pandas is not installed. Adding to installation list...")
    missing_packages.append("pandas")

# Install any missing packages
if missing_packages:
    print(f"Installing packages: {', '.join(missing_packages)}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
    print("All required packages installed successfully.")

# Import TensorFlow (we're using MobileNetV2 directly from keras, no need for TF Hub)
import tensorflow as tf

print(f"Using TensorFlow version: {tf.__version__}")
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add the project root to the path
current_dir = pathlib.Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

# Import TinyLCM utilities for compatibility testing
from examples.utils.preprocessors import (prepare_input_tensor_quantized,
                                          resize_image)

# Configuration
IMAGE_SIZE = (224, 224)  # MobileNetV2 default input size
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2
CLASS_MODE = 'categorical'  # One-hot encoded labels

# Directories
IMAGE_DIR = os.path.join(root_dir, "examples/assets/training_images")  # Aktualisiert zum neuen Pfad
OUTPUT_DIR = os.path.join(root_dir, "examples/scenario2_drift_objects/model")  # Direkt ins Szenario-Verzeichnis
CSV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "training_metrics")  # Unterverzeichnis für CSV-Daten
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

# Output file paths
MODEL_KERAS_PATH = os.path.join(OUTPUT_DIR, "model_object.h5")  # Vereinfachte Namen
MODEL_TFLITE_PATH = os.path.join(OUTPUT_DIR, "model_object.tflite")
LABELS_FILE_PATH = os.path.join(OUTPUT_DIR, "labels_object.txt")

# Wir brauchen keine Custom-Pfade mehr, da wir direkt ins Szenario-Verzeichnis speichern
TINYML_MODEL_PATH = MODEL_TFLITE_PATH  # Verwende den gleichen Pfad
TINYML_LABELS_PATH = LABELS_FILE_PATH  # Verwende den gleichen Pfad

def create_data_generators():
    """Create training and validation data generators."""
    # Check that the image directory exists
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
    
    # List available classes (subdirectories)
    class_dirs = [d for d in os.listdir(IMAGE_DIR) 
                  if os.path.isdir(os.path.join(IMAGE_DIR, d))]
    
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {IMAGE_DIR}")
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    # Create class to index mapping
    class_indices = {class_name: i for i, class_name in enumerate(sorted(class_dirs))}
    
    # Save class mapping to labels file
    with open(LABELS_FILE_PATH, 'w') as f:
        for class_name, idx in class_indices.items():
            f.write(f"{idx} {class_name}\n")
    
    print(f"Saved class labels to {LABELS_FILE_PATH}")
    
    # Create data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        IMAGE_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset='training',
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        IMAGE_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset='validation',
        shuffle=False,
        seed=RANDOM_SEED
    )
    
    # Verify that data generators are working
    print(f"Training generator found {train_generator.samples} samples")
    print(f"Validation generator found {val_generator.samples} samples")
    
    return train_generator, val_generator, len(class_dirs)

def create_transfer_learning_model(num_classes):
    """Create a transfer learning model with MobileNetV2."""
    print(f"Creating model with MobileNetV2 from Keras applications...")
    
    # Create the MobileNetV2 base model without top layers
    # MobileNetV2 is optimized for mobile and edge devices like Raspberry Pi Zero
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Create input layer
    inputs = base_model.input
    
    # Add global pooling layer to get feature vector
    x = base_model.output
    base_features = GlobalAveragePooling2D(name='feature_pool')(x)
    
    print(f"Successfully loaded MobileNetV2 model")
    print(f"Base model output shape before pooling: {x.shape}")
    print(f"Feature vector dimension after pooling: {base_features.shape}")
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom classification layers on top of the features
    # First create a rich feature vector layer that will be useful for KNN and drift detection
    feature_vector = Dense(512, activation='relu', name='feature_vector')(base_features)
    x = Dropout(0.5)(feature_vector)
    output = Dense(num_classes, activation='softmax', name='class_output')(x)
    
    # Create models - one for training and one with multiple outputs for extraction
    # Regular training model
    model = Model(inputs=inputs, outputs=output)
    
    # Create a separate feature extraction model with multiple outputs
    # This allows us to extract both the feature vector and classification in one pass
    feature_extraction_model = Model(
        inputs=inputs,
        outputs={
            'features': feature_vector,  # Accessible 512-d feature vector
            'classes': output            # Class predictions
        }
    )
    
    # Also create a model that extracts just the raw features
    # This is extremely useful for drift detection
    raw_feature_model = Model(
        inputs=inputs,
        outputs=base_features
    )
    
    print(f"Model creation complete:")
    print(f"- Base model produces {int(base_features.shape[1])}D features")
    print(f"- Custom feature layer produces {int(feature_vector.shape[1])}D features")
    print(f"- Model outputs {num_classes} classes")
    
    return model, feature_extraction_model, raw_feature_model

def train_model(model, train_generator, val_generator):
    """Train the transfer learning model."""
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = val_generator.samples // BATCH_SIZE
    
    print("Starting model training...")
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # Speichere Trainingshistorie als CSV
    if hasattr(history, 'history'):
        save_history_to_csv(history.history, "initial_training")
    
    return history

def save_history_to_csv(history_dict, prefix="training"):
    """
    Speichert die Trainingshistorie als CSV-Datei.
    
    Args:
        history_dict: Dictionary mit den Trainingsmetriken
        prefix: Präfix für den Dateinamen
    """
    try:
        # Importiere pandas lokal, falls der globale Import fehlgeschlagen ist
        try:
            import pandas as pd
        except ImportError:
            print("Pandas konnte nicht importiert werden. CSV-Export wird übersprungen.")
            return
            
        # Erstelle einen DataFrame aus dem History-Dictionary
        df = pd.DataFrame(history_dict)
        
        # Füge eine Epochenspalte hinzu
        df.insert(0, 'epoch', range(1, len(df) + 1))
        
        # Speichere als CSV
        csv_file = os.path.join(CSV_OUTPUT_DIR, f"{prefix}_history.csv")
        df.to_csv(csv_file, index=False)
        print(f"Trainingshistorie gespeichert in: {csv_file}")
        
        # Speichere auch Zusammenfassung der finalen Metriken
        final_metrics = {metric: values[-1] for metric, values in history_dict.items()}
        final_df = pd.DataFrame([final_metrics])
        final_df['training_type'] = prefix
        
        final_csv = os.path.join(CSV_OUTPUT_DIR, f"{prefix}_final_metrics.csv")
        final_df.to_csv(final_csv, index=False)
        print(f"Finale Metriken gespeichert in: {final_csv}")
        
    except Exception as e:
        print(f"Fehler beim Speichern der Trainingshistorie: {e}")
        import traceback
        traceback.print_exc()

def fine_tune_model(model, train_generator, val_generator):
    """Fine-tune the model with a conservative approach (only training the top layers)."""
    # For a conservative approach, we only train the classification head
    # We keep the base model frozen to prevent overfitting
    
    # Uncomment the following section if you want more aggressive fine-tuning:
    """
    # Find the base model (MobileNetV2) layer
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # This is likely our base model
            # Unfreeze the last few layers of the found base model (very conservative)
            for sublayer in layer.layers[-5:]:  # Reduced from 20 to 5 layers
                sublayer.trainable = True
            break
    """
    
    # Set only the top layers to be trainable
    trainable_found = False
    for layer in model.layers[-2:]:  # Only unfreeze the last classification layers
        layer.trainable = True
        trainable_found = True
        print(f"Set layer '{layer.name}' to trainable")

    if not trainable_found:
        print("Warning: No layers were set to trainable for fine-tuning")
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = val_generator.samples // BATCH_SIZE
    
    print("Starting model fine-tuning...")
    
    # Fine-tune the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS // 2,  # Fewer epochs for fine-tuning
        validation_data=val_generator,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # Speichere Trainingshistorie als CSV
    if hasattr(history, 'history'):
        save_history_to_csv(history.history, "fine_tuning")
    
    return history

def convert_to_tflite(feature_extraction_model, raw_feature_model, quantize=True):
    """Convert the model to TFLite format with optional quantization."""
    # First save the raw feature model - this is the one we'll use for TinyLCM feature extraction
    raw_feature_model.save(MODEL_KERAS_PATH.replace('.h5', '_features.h5'))
    print(f"Saved feature extractor model to {MODEL_KERAS_PATH.replace('.h5', '_features.h5')}")
    
    print("\n==== Converting MobileNetV2 Model to TFLite ====")
    # Get the feature dimension
    output_shape = raw_feature_model.output.shape
    feature_dim = output_shape[-1] if len(output_shape) >= 2 else output_shape[0]
    print(f"This model will provide rich features (~{feature_dim} dimensions) for drift detection")
    
    # Convert specifically the raw_feature_model that extracts the features
    # This is critical for drift detection quality
    converter = tf.lite.TFLiteConverter.from_keras_model(raw_feature_model)
    
    # Configure the converter for optimal edge performance
    if quantize:
        print("Applying int8 quantization to reduce model size and increase performance...")
        # Enable full integer quantization for best performance on Pi Zero
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configure int8 quantization
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,  # Int8 operations for CPU
            tf.lite.OpsSet.TFLITE_BUILTINS        # Regular operations as fallback
        ]
        
        # Set the inference input/output type
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32  # Keep output as float32 for feature quality
        
        # Representative dataset for calibration (use a subset of validation data)
        def representative_data_gen():
            num_calibration_examples = 100
            print(f"Generating calibration data from {num_calibration_examples} examples...")
            count = 0
            
            for input_value, _ in val_generator:
                # Only use a subset of the data
                if count >= num_calibration_examples:
                    break
                    
                # Batch size might be smaller than expected at the end
                for i in range(min(BATCH_SIZE, input_value.shape[0])):
                    if count >= num_calibration_examples:
                        break
                    
                    # Get a single image
                    image = input_value[i:i+1]
                    
                    # Yield the image for calibration
                    yield [image]
                    count += 1
                    
                    if count % 10 == 0:
                        print(f"  Processed {count}/{num_calibration_examples} calibration examples")
                    
        # Set the representative dataset
        converter.representative_dataset = representative_data_gen
    else:
        print("Skipping quantization (model will be larger)")
    
    print("Converting model to TFLite format...")
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(MODEL_TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    # Since we're saving directly to the scenario directory, the path is the same
    model_size_mb = os.path.getsize(MODEL_TFLITE_PATH) / (1024 * 1024)
    print(f"✅ TFLite model saved to: {MODEL_TFLITE_PATH}")
    print(f"   Model size: {model_size_mb:.2f} MB")
    
    # Also save the labels file
    with open(LABELS_FILE_PATH, 'w') as f:
        for i, class_name in enumerate(sorted(train_generator.class_indices.keys())):
            f.write(f"{i} {class_name}\n")
    
    print(f"✅ Labels file saved to: {LABELS_FILE_PATH}")
    
    print("\nQuantized model is optimized for Raspberry Pi Zero and provides:")
    print(f"1. Higher feature dimensionality ({feature_dim}D) - excellent for drift detection")
    print("2. Reduced model size through int8 quantization")
    print("3. Faster inference on CPU")
    
    return tflite_model

def save_pca_stats_to_csv(pca, explained_variance_ratio, cumulative_explained_variance, prefix="pca"):
    """
    Speichert PCA-Statistiken als CSV-Dateien.
    
    Args:
        pca: Trainiertes PCA-Objekt
        explained_variance_ratio: Array mit der erklärten Varianz pro Komponente
        cumulative_explained_variance: Array mit der kumulierten erklärten Varianz
        prefix: Präfix für den Dateinamen
    """
    try:
        # Importiere pandas - sollte zu diesem Zeitpunkt bereits installiert sein
        import pandas as pd
        
        # Erstelle DataFrame für die erklärte Varianz pro Komponente
        variance_data = {
            "component_index": np.arange(1, len(explained_variance_ratio) + 1),
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_explained_variance": cumulative_explained_variance
        }
        
        variance_df = pd.DataFrame(variance_data)
        variance_csv = os.path.join(CSV_OUTPUT_DIR, f"{prefix}_variance.csv")
        variance_df.to_csv(variance_csv, index=False)
        print(f"PCA Varianz-Statistiken gespeichert in: {variance_csv}")
        
        # Speichere die PCA-Komponenten
        if hasattr(pca, 'components_'):
            # Diese können sehr groß sein, daher nur speichern, wenn gewünscht
            save_components = True
            if save_components:
                components_df = pd.DataFrame(pca.components_)
                components_csv = os.path.join(CSV_OUTPUT_DIR, f"{prefix}_components.csv")
                components_df.to_csv(components_csv, index=False, header=False)
                print(f"PCA Komponenten gespeichert in: {components_csv}")
        
    except Exception as e:
        print(f"Fehler beim Speichern der PCA-Statistiken: {e}")
        import traceback
        traceback.print_exc()

def train_and_save_feature_processor(raw_feature_model, train_generator, output_dir, pca_components=256):
    """
    Trainiert und speichert einen Feature-Prozessor, der StandardScaler und PCA kombiniert.
    
    Args:
        raw_feature_model: Das Modell, das die Features extrahiert
        train_generator: Der Daten-Generator für die Trainingsbilder
        output_dir: Das Verzeichnis, in dem der Feature-Prozessor gespeichert wird
        pca_components: Anzahl der PCA-Komponenten (Default: 256)
        
    Returns:
        Tuple aus (StandardScaler, PCA, Pfad zur gespeicherten Datei)
    """
    try:
        # Import der benötigten Bibliotheken
        import pickle
        import time

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("\n❌ ERROR: scikit-learn not installed. Installing required packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
        import pickle
        import time

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    
    print(f"\n==== Training Feature Processor (StandardScaler + PCA) ====")
    
    # Sammle Features aus den Trainingsdaten
    print("Collecting features from training data...")
    features_list = []
    sample_count = 0
    
    # Setze den Generator zurück, falls möglich
    if hasattr(train_generator, 'reset'):
        train_generator.reset()
    
    # Durchlaufe den Generator komplett für alle verfügbaren Samples
    # Da der Datensatz klein ist, nutzen wir alle verfügbaren Daten
    start_time = time.time()
    num_total_train_samples = train_generator.samples
    samples_processed_in_loop = 0 # Zähler für tatsächlich verarbeitete Samples

    # Iteriere über die Batches, aber stoppe nach train_generator.samples
    for batch_idx in range((num_total_train_samples + train_generator.batch_size - 1) // train_generator.batch_size):
        if samples_processed_in_loop >= num_total_train_samples:
            break # Stoppe, wenn alle Samples gesehen wurden

        inputs, _ = next(train_generator) # Hole nächsten Batch
        batch_features = raw_feature_model.predict(inputs, verbose=0)

        for feature_idx in range(batch_features.shape[0]):
            if samples_processed_in_loop < num_total_train_samples:
                flat_feature = batch_features[feature_idx].flatten() if hasattr(batch_features[feature_idx], 'flatten') else batch_features[feature_idx]
                features_list.append(flat_feature)
                samples_processed_in_loop += 1
            else:
                break # Innerer Loop stoppen
        
        sample_count = samples_processed_in_loop # sample_count für Logging aktualisieren
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0 or sample_count == num_total_train_samples:
            print(f"  Processed {sample_count}/{num_total_train_samples} samples ({len(features_list)} features collected)")
    
    elapsed = time.time() - start_time
    print(f"Feature collection complete in {elapsed:.2f} seconds")
    
    # Überprüfe, ob Features gesammelt wurden
    if not features_list:
        print("❌ ERROR: No features collected! Check the train_generator and model.")
        return None, None, None
        
    # Konvertiere zu NumPy-Array
    try:
        features_array = np.array(features_list)
        print(f"Collected {len(features_list)} feature vectors of dimension {features_array.shape[1]}")
    except Exception as e:
        print(f"❌ ERROR converting features to array: {e}")
        return None, None, None
    
    # 1. Standardisierung (Mittelwert 0, Std 1)
    print("Standardizing features...")
    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features_array)
        print(f"✓ Features standardized (mean ≈ 0, std ≈ 1)")
    except Exception as e:
        print(f"❌ ERROR in standardization: {e}")
        # Speichere nur den Scaler als Fallback
        processor_path = os.path.join(output_dir, "feature_processor.pkl")
        with open(processor_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'pca': None}, f)
        print(f"⚠️ Only StandardScaler saved (without PCA) to: {processor_path}")
        return scaler, None, processor_path
    
    # 2. PCA-Anwendung mit Sicherheitsabfrage für die Komponentenanzahl
    # Stelle sicher, dass nicht mehr Komponenten als Samples oder Features angefordert werden
    max_possible_components = min(scaled_features.shape[0], scaled_features.shape[1])
    actual_pca_components = min(pca_components, max_possible_components - 1)
    
    if actual_pca_components < pca_components:
        print(f"⚠️ WARNING: Reduced PCA components from {pca_components} to {actual_pca_components}")
        print(f"   (limited by available samples or feature dimensions)")
    
    try:
        print(f"Fitting PCA to reduce dimensions from {features_array.shape[1]} to {actual_pca_components}...")
        pca = PCA(n_components=actual_pca_components)
        pca.fit(scaled_features)
        
        # Test der Transformation
        test_features = scaled_features[:1]
        reduced_features = pca.transform(test_features)
        
        # Erklärte Varianz berechnen
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        explained_variance = sum(explained_variance_ratio) * 100
        
        # Speichere PCA-Statistiken als CSV
        save_pca_stats_to_csv(
            pca=pca, 
            explained_variance_ratio=explained_variance_ratio,
            cumulative_explained_variance=cumulative_explained_variance,
            prefix="feature_processor_pca"
        )
        
        # Speichere Feature-Stichproben als CSV
        sample_size = min(100, len(scaled_features))
        if sample_size > 0:
            # Original Features (skaliert)
            orig_sample_df = pd.DataFrame(scaled_features[:sample_size])
            orig_sample_csv = os.path.join(CSV_OUTPUT_DIR, "features_original_sample.csv")
            orig_sample_df.to_csv(orig_sample_csv, index=False, header=False)
            
            # PCA-reduzierte Features
            pca_sample = pca.transform(scaled_features[:sample_size])
            pca_sample_df = pd.DataFrame(pca_sample)
            pca_sample_csv = os.path.join(CSV_OUTPUT_DIR, "features_pca_reduced_sample.csv")
            pca_sample_df.to_csv(pca_sample_csv, index=False, header=False)
            
            print(f"Feature-Samples exportiert (original und PCA-reduziert)")
            
        print(f"✅ PCA successfully trained")
        print(f"   - Explains {explained_variance:.2f}% of total variance")
        print(f"   - Reduced dimensions: {features_array.shape[1]} → {actual_pca_components}")
        print(f"   - Test transform shape: {test_features.shape} → {reduced_features.shape}")
    except Exception as e:
        print(f"❌ ERROR in PCA fitting: {e}")
        # Speichere nur den Scaler als Fallback
        processor_path = os.path.join(output_dir, "feature_processor.pkl")
        with open(processor_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'pca': None}, f)
        print(f"⚠️ Only StandardScaler saved (without PCA) to: {processor_path}")
        return scaler, None, processor_path
    
    # Beide Modelle (Scaler und PCA) speichern
    processor_path = os.path.join(output_dir, "feature_processor.pkl")
    try:
        with open(processor_path, 'wb') as f:
            processor = {
                'scaler': scaler, 
                'pca': pca,
                'input_dim': features_array.shape[1],
                'output_dim': actual_pca_components,
                'explained_variance': explained_variance,
                'creation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            pickle.dump(processor, f)
        
        print(f"✅ Feature processor (StandardScaler + PCA) saved to: {processor_path}")
    except Exception as e:
        print(f"❌ ERROR saving feature processor: {e}")
        return scaler, pca, None
    
    # Zusätzliche Debug-Informationen
    print("\nFeature Statistics:")
    print(f"- Original features range: [{features_array.min():.4f}, {features_array.max():.4f}]")
    print(f"- Scaled features range: [{scaled_features.min():.4f}, {scaled_features.max():.4f}]")
    print(f"- Reduced features range: [{reduced_features.min():.4f}, {reduced_features.max():.4f}]")
    
    # Erstelle Beispielcode für die Verwendung
    example_code = f"""
# Beispielcode zur Verwendung des Feature-Prozessors in TinyLCM:

import pickle
import numpy as np

# Feature-Prozessor laden
with open("{os.path.basename(processor_path)}", 'rb') as f:
    processor = pickle.load(f)

scaler = processor['scaler']
pca = processor['pca']
input_dim = processor['input_dim']
output_dim = processor['output_dim']

# In der Pipeline nach Feature-Extraktion:
def process_features(features):
    # Flatten, falls nötig
    if len(features.shape) > 1:
        features = features.flatten()
        
    # Dimensionscheck
    if len(features) != input_dim:
        # Hier ggf. Fehlerbehandlung oder Anpassung
        pass
        
    # Standardisierung und PCA anwenden
    scaled = scaler.transform(features.reshape(1, -1))
    reduced = pca.transform(scaled)[0]
    
    # Reduzierte Features (Dimension: {actual_pca_components}) zurückgeben
    return reduced
"""
    
    # Speichere Beispielcode
    example_path = os.path.join(output_dir, "feature_processor_usage.py")
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    return scaler, pca, processor_path

def save_model_metadata_to_csv(input_shape, output_shape, input_type, output_type):
    """
    Speichert Modell-Metadaten als CSV-Datei.
    """
    try:
        # Importiere pandas - sollte zu diesem Zeitpunkt bereits installiert sein
        import pandas as pd
        import datetime
        
        # Erstelle DataFrame für die Modell-Metadaten
        metadata = [{
            'model_name': os.path.basename(MODEL_TFLITE_PATH),
            'model_type': 'MobileNetV2',
            'input_shape': str(input_shape),
            'output_shape': str(output_shape),
            'input_type': str(input_type),
            'output_type': str(output_type),
            'timestamp': datetime.datetime.now().isoformat()
        }]
        
        metadata_df = pd.DataFrame(metadata)
        metadata_csv = os.path.join(CSV_OUTPUT_DIR, "model_metadata.csv")
        metadata_df.to_csv(metadata_csv, index=False)
        print(f"Modell-Metadaten gespeichert in: {metadata_csv}")
        
    except Exception as e:
        print(f"Fehler beim Speichern der Modell-Metadaten: {e}")
        import traceback
        traceback.print_exc()

def test_tflite_compatibility():
    """Test that the TFLite model is compatible with TinyLCM preprocessors and provides rich features."""
    print("\n==== Testing TFLite Model Compatibility with TinyLCM ====")
    
    # Load the TFLite model from the correct path
    print(f"Loading TFLite model from: {MODEL_TFLITE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print("\nModel specifications:")
    print(f"Input shape: {input_details['shape']}")
    print(f"Input type: {input_details['dtype']}")
    print(f"Output shape: {output_details['shape']}")
    print(f"Output type: {output_details['dtype']}")
    
    # Check all output tensor details
    all_output_details = interpreter.get_output_details()
    print(f"\nModel has {len(all_output_details)} output tensors:")
    for i, output in enumerate(all_output_details):
        print(f"  Output {i}: shape={output['shape']}, name={output.get('name', 'unnamed')}")
    
    # Create a test image for inference
    print("\nRunning test inference...")
    test_image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Test with TinyLCM preprocessor
    try:
        # Preprocess using TinyLCM utilities to ensure compatibility
        processed_image = prepare_input_tensor_quantized(test_image, input_details)
        print(f"Successfully preprocessed image with TinyLCM utilities")
        
        # Set the input tensor
        interpreter.set_tensor(input_details['index'], processed_image)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output (feature vector)
        output = interpreter.get_tensor(output_details['index'])
        
        # Check output format and dimensionality
        feature_size = output.size
        print(f"\nFeature vector details:")
        print(f"- Shape: {output.shape}")
        print(f"- Total dimensions: {feature_size}")
        print(f"- Value range: min={output.min():.5f}, max={output.max():.5f}")
        
        # Speichere Modell-Metadaten als CSV
        save_model_metadata_to_csv(
            input_shape=input_details['shape'],
            output_shape=output_details['shape'],
            input_type=input_details['dtype'],
            output_type=output_details['dtype']
        )
        
        # Speichere eine Stichprobe der Feature-Vektoren für spätere Visualisierung
        try:
            # Erzeugen von 10 Test-Tensoren mit leicht variiertem Rauschen
            test_images = []
            for i in range(10):
                # Erzeuge Bild mit leicht variiertem Rauschen
                random_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                test_images.append(random_img)
                
            # Extrahiere Features aus den Test-Bildern
            feature_vectors = []
            for img in test_images:
                processed_image = prepare_input_tensor_quantized(img, input_details)
                interpreter.set_tensor(input_details['index'], processed_image)
                interpreter.invoke()
                feat_vec = interpreter.get_tensor(output_details['index']).flatten()
                feature_vectors.append(feat_vec)
                
            # Speichere als CSV
            if feature_vectors:
                feature_df = pd.DataFrame(feature_vectors)
                feature_csv = os.path.join(CSV_OUTPUT_DIR, "test_feature_vectors.csv")
                feature_df.to_csv(feature_csv, index=False, header=False)
                print(f"Test-Feature-Vektoren gespeichert in: {feature_csv}")
        except Exception as e:
            print(f"Fehler beim Exportieren der Test-Feature-Vektoren: {e}")
        
        # Evaluate feature quality for drift detection
        if feature_size >= 1000:
            print(f"\n✅ EXCELLENT! Feature vector has {feature_size} dimensions")
            print("   This is ideal for high-quality drift detection and KNN classification")
            print("   Drift detection will be very sensitive to subtle changes in input distribution")
        elif feature_size >= 256:
            print(f"\n✅ GOOD! Feature vector has {feature_size} dimensions")
            print("   This should provide good drift detection capabilities")
        elif feature_size >= 64:
            print(f"\n✓ ACCEPTABLE: Feature vector has {feature_size} dimensions")
            print("   This can work for drift detection but might miss subtle changes")
        else:
            print(f"\n⚠️ WARNING: Feature vector has only {feature_size} dimensions")
            print("   This is not ideal for effective drift detection")
            print("   Consider using a different feature layer or model architecture")
        
        # Print sample of the feature vector
        print("\nFeature vector sample (first 10 values):")
        print(f"  {output.flatten()[:10]}")
        
        print("\n✅ TFLite model successfully tested and is compatible with TinyLCM!")
        
    except Exception as e:
        print("\n❌ Error testing TFLite compatibility:")
        print(str(e))
        
    # Now explicitly try creating a KNN state with this model to verify
    try:
        # Import LightweightKNN to check state creation
        import time  # Import time for timestamps

        from tinylcm.core.classifiers.knn import LightweightKNN

        # Instantiate a KNN with 4 classes
        knn = LightweightKNN(
            k=5, 
            max_samples=160, 
            distance_metric="euclidean",
            use_numpy=True
        )
        
        # Test feature extraction and state saving
        print("Testing KNN state creation...")
        num_samples = 4  # Just for testing
        
        # Bestimme ob die Features mehr als nur 4 Dimensionen haben (ideal wäre > 100)
        num_features = output.flatten().shape[0]  # Use the actual feature dimension
        
        if num_features <= 4:
            # Zeige eine deutliche Warnung an
            print("\n⚠️ WARNING: Feature dimension is only 4! ⚠️")
            print("This will severely limit drift detection capability.")
            print("Try with a different feature layer index (e.g., -2, -3, or -4)")
            print("Using feature_layer_index=-3 in the config file is recommended.\n")
        elif num_features > 100:
            print(f"\n✅ EXCELLENT! Feature dimension is {num_features}, which is ideal for drift detection.\n")
        else:
            print(f"\n✓ Feature dimension is {num_features}. Acceptable, but more would be better.\n")
        
        # Create synthetic features and labels
        features = np.random.rand(num_samples, num_features)
        labels = ["negative", "lego", "stone", "tire"]
        timestamps = [time.time() - i*10 for i in range(num_samples)]
        
        # Fit the KNN
        knn.fit(features, labels, timestamps)
        
        # Get state
        state = knn.get_state()
        
        # Speichere den KNN-Zustand für Tests in einer temporären Datei
        initial_state_path = os.path.join(OUTPUT_DIR, "test_knn_state.json")
        with open(initial_state_path, 'w') as f:
            import json
            json.dump({
                "classifier": state,
                "metadata": {
                    "description": "Test KNN state for model verification",
                    "creation_date_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "feature_dimension": num_features,
                    "source_model": MODEL_TFLITE_PATH
                }
            }, f, indent=4)
        
        # Exportiere KNN Test-Zustand als CSV
        try:
            # Erstelle einen DataFrame mit Features und Labels
            knn_test_df = pd.DataFrame(features)
            knn_test_df['label'] = labels
            knn_test_df['timestamp'] = timestamps
            
            # Speichere als CSV
            knn_test_csv = os.path.join(CSV_OUTPUT_DIR, "knn_test_features.csv")
            knn_test_df.to_csv(knn_test_csv, index=False)
            print(f"KNN Test-Features gespeichert in: {knn_test_csv}")
        except Exception as e:
            print(f"Fehler beim Exportieren der KNN Test-Features: {e}")
            
        print(f"✅ Successfully created KNN state with {num_features}-dimensional features")
        print(f"Test state saved to: {initial_state_path}")
        print(f"State keys: {list(state.keys())}")
        
    except Exception as e:
        print("❌ Error testing KNN state creation:")
        print(str(e))

def update_config_files():
    """Update the config files to use the new TFLite model."""
    # Update config_scenario2.json to use the new model
    config_file = os.path.join(
        root_dir, 
        "examples/scenario2_drift_objects/config_scenario2.json"
    )
    
    if os.path.exists(config_file):
        import json

        # Relative paths from the config file to the model files
        # Da wir jetzt direkt ins scenario2/model Verzeichnis speichern, können wir relative Pfade verwenden
        rel_model_path = "./model/model_object.tflite"
        rel_labels_path = "./model/labels_object.txt"
        
        # Read existing config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Update model paths
        config["model"]["model_path"] = rel_model_path
        config["model"]["labels_path"] = rel_labels_path
        config["tinylcm"]["feature_extractor"]["model_path"] = rel_model_path
        
        # Using layer_index 0 to access the properly pooled MobileNetV2 features
        # This gives us a 1280-dimensional feature vector (similar to EfficientNet-Lite0)
        config["tinylcm"]["feature_extractor"]["feature_layer_index"] = 0
        
        # Erhöhe die maximale Anzahl an Samples im KNN für bessere Genauigkeit
        if "adaptive_classifier" in config["tinylcm"]:
            config["tinylcm"]["adaptive_classifier"]["max_samples"] = 200
        
        # Schreibe die aktualisierte Konfiguration
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Updated config file: {config_file}")
        print(f"   - Set model_path to {rel_model_path}")
        print(f"   - Set labels_path to {rel_labels_path}")
        print(f"   - Set feature_layer_index to 0 (MobileNetV2 Feature Vector)")
        print(f"   - Increased KNN max_samples to 200 for better accuracy")
    else:
        print(f"\n⚠️ Config file not found: {config_file}")

if __name__ == "__main__":
    print("Starting transfer learning for TinyLCM Object Classification")
    print(f"Using images from: {IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Create data generators
    train_generator, val_generator, num_classes = create_data_generators()
    
    # Create models - standard training model and feature extraction model
    model, feature_extraction_model, raw_feature_model = create_transfer_learning_model(num_classes)
    print("Training model architecture:")
    print(model.summary())
    
    print("\nFeature extraction model architecture:")
    for name, output in feature_extraction_model.output.items():
        print(f"Output '{name}': shape={output.shape}")
        
    print("\nRaw feature model architecture:")
    outputs = raw_feature_model.output
    print(f"Raw features shape: {outputs.shape}")
    
    # Train the model
    history = train_model(model, train_generator, val_generator)
    
    # Fine-tune the model
    history = fine_tune_model(model, train_generator, val_generator)
    
    # Convert to TFLite format
    tflite_model = convert_to_tflite(feature_extraction_model, raw_feature_model, quantize=True)
    
    # Test compatibility with TinyLCM
    test_tflite_compatibility()
    
    # Update config files to use the new model
    update_config_files()
    
    # Train and save the feature processor for improved performance on Pi Zero
    print("\nTraining feature processor for dimension reduction...")
    try:
        scaler, pca, processor_path = train_and_save_feature_processor(
            raw_feature_model, 
            train_generator,  # Immer die Trainingsdaten verwenden, um Data Leakage zu vermeiden
            OUTPUT_DIR, 
            pca_components=256  # Reduziere auf 256 Dimensionen für eine gute Balance
        )
        
        # Aktualisiere die Konfigurationsdatei, um den Feature-Prozessor zu verwenden
        if processor_path:
            # Füge Feature-Prozessor zur Konfiguration hinzu
            config_file = os.path.join(root_dir, "examples/scenario2_drift_objects/config_scenario2.json")
            if os.path.exists(config_file):
                import json

                # Lese bestehende Konfiguration
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Füge Feature-Prozessor-Konfiguration hinzu
                config["tinylcm"]["feature_processor"] = {
                    "enabled": True,
                    "model_path": "./model/feature_processor.pkl"
                }
                
                # Schreibe aktualisierte Konfiguration
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"\n✅ Updated config file with feature processor configuration")
                print(f"   Original features: 1280D → Reduced features: 256D")
                print(f"   This will significantly improve KNN performance on the Pi Zero")
        else:
            print(f"\n⚠️ Failed to create feature processor, continuing without dimension reduction")
    except Exception as e:
        print(f"\n❌ ERROR during feature processor training: {e}")
        print(f"   Continuing without dimension reduction")
    
    print("\n===============================================")
    print("✅ MobileNetV2 MODEL TRAINING COMPLETE")
    print("===============================================")
    print(f"Model saved to: {MODEL_TFLITE_PATH}")
    print(f"Labels saved to: {LABELS_FILE_PATH}")
    
    print("\nFeatures:")
    print("✓ Optimized for Raspberry Pi Zero 2W")
    print("✓ ~1280-dimensional feature vectors from MobileNetV2 for excellent drift detection")
    print("✓ StandardScaler + PCA dimension reduction to 256D for faster KNN calculations")
    print("✓ Int8 quantized for efficient inference")
    print("✓ Directly compatible with TinyLCM")
    
    print("\nTo use the model:")
    print("1. Run the initial state creation script:")
    print("   cd examples/scenario2_drift_objects/initial_state")
    print("   python create_inital_knn_sc2.py")
    
    print("\n2. Run the main scenario script:")
    print("   cd ..")
    print("   python main_scenario2.py")
    
    print("\nThis optimized MobileNetV2 model with StandardScaler + PCA dimension reduction")
    print("will significantly improve both drift detection quality and performance")
    print("compared to the previous 4-dimensional model output.")