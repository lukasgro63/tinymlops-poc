#!/usr/bin/env python3
"""
TensorFlow Transfer Learning for TinyLCM Object Classification

This script trains a transfer learning model for object classification using
images from examples/assets/initial_states/images. It creates a model that:
1. Uses MobileNetV2 as the base model (efficient for edge devices)
2. Adds custom classification layers
3. Extracts intermediate features from the penultimate layer
4. Exports to TFLite format with quantization
5. Is compatible with the TinyLCM pipeline

The model is saved in a format compatible with the preprocessors.py and main_scenario2.py files.
"""

import os
import sys
import numpy as np
import pathlib
import subprocess

# Check for required dependencies and install if missing
required_packages = ["Pillow", "scipy"]
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

# Install any missing packages
if missing_packages:
    print(f"Installing missing packages: {', '.join(missing_packages)}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
    print("All required packages installed successfully.")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add the project root to the path
current_dir = pathlib.Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

# Import TinyLCM utilities for compatibility testing
from examples.utils.preprocessors import prepare_input_tensor_quantized, resize_image

# Configuration
IMAGE_SIZE = (224, 224)  # MobileNetV2 default input size
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2
CLASS_MODE = 'categorical'  # One-hot encoded labels

# Directories
IMAGE_DIR = os.path.join(root_dir, "examples/assets/initial_states/images")
OUTPUT_DIR = os.path.join(root_dir, "examples/assets/model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file paths
MODEL_KERAS_PATH = os.path.join(OUTPUT_DIR, "model_object_transfer.h5")
MODEL_TFLITE_PATH = os.path.join(OUTPUT_DIR, "model_object_transfer.tflite")
LABELS_FILE_PATH = os.path.join(OUTPUT_DIR, "labels_object_transfer.txt")

# Custom model paths for TinyLCM compatibility
TINYML_MODEL_PATH = os.path.join(OUTPUT_DIR, "model_object_1.tflite")  # This name should match what's used in configs
TINYML_LABELS_PATH = os.path.join(OUTPUT_DIR, "labels_object_1.txt")   # This name should match what's used in configs

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
    # Create base model with pre-trained weights
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create feature extraction layer that we will later use
    # Important: We want the 'out_relu' layer for the rich 1280-dimensional feature vector
    feature_layer = base_model.get_layer('out_relu')
    
    # Create a clean output from the feature layer for later extraction
    feature_output = feature_layer.output
    
    # Add custom classification layers on top of the features
    x = GlobalAveragePooling2D()(feature_output)
    feature_vector = Dense(1024, activation='relu', name='feature_vector')(x)
    x = Dropout(0.5)(feature_vector)
    output = Dense(num_classes, activation='softmax', name='class_output')(x)
    
    # Create models - one for training and one with multiple outputs for extraction
    # Regular training model
    model = Model(inputs=base_model.input, outputs=output)
    
    # Create a separate feature extraction model with multiple outputs
    # This allows us to extract both the raw features and classification in one pass
    feature_extraction_model = Model(
        inputs=base_model.input,
        outputs={
            'features': feature_vector,  # Accessible 1024-d feature vector
            'classes': output            # Class predictions
        }
    )
    
    # For debugging purposes, also create a model that extracts the raw penultimate layer
    raw_feature_model = Model(
        inputs=base_model.input,
        outputs=feature_layer.output
    )
    
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
    
    return history

def fine_tune_model(model, train_generator, val_generator):
    """Fine-tune the model by unfreezing some layers."""
    # Unfreeze some layers for fine-tuning
    # Find the base model (MobileNetV2) layer
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # This is likely our base model
            # Unfreeze the last 20 layers of the found base model
            for sublayer in layer.layers[-20:]:
                sublayer.trainable = True
            break
    
    # If no base model found, just unfreeze the last few layers of the main model
    trainable_found = False
    for layer in model.layers[-5:]:  # Try last 5 layers
        layer.trainable = True
        trainable_found = True
    
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
    
    return history

def convert_to_tflite(feature_extraction_model, raw_feature_model, quantize=True):
    """Convert the model to TFLite format with optional quantization."""
    # First save the raw feature model - this is the one we'll use for TinyLCM feature extraction
    raw_feature_model.save(MODEL_KERAS_PATH.replace('.h5', '_features.h5'))
    print(f"Saved feature extractor model to {MODEL_KERAS_PATH.replace('.h5', '_features.h5')}")
    
    # Now convert the feature extraction model to TFLite
    # Create a TensorFlow Lite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(raw_feature_model)
    
    if quantize:
        # Post-training quantization to reduce model size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.uint8]
        
        # Representative dataset for calibration (use a subset of validation data)
        def representative_data_gen():
            num_calibration_examples = 100
            for input_value, _ in val_generator:
                # Only use a subset of the data
                if num_calibration_examples <= 0:
                    break
                # Batch size might be smaller than expected at the end
                for i in range(min(BATCH_SIZE, input_value.shape[0])):
                    if num_calibration_examples <= 0:
                        break
                    image = input_value[i:i+1]
                    yield [image]
                    num_calibration_examples -= 1
                    
        converter.representative_dataset = representative_data_gen
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TFLite model under both paths
    with open(MODEL_TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    # Copy to standard TinyLCM path for compatibility
    with open(TINYML_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved TFLite models to:")
    print(f"  - {MODEL_TFLITE_PATH} (for reference)")
    print(f"  - {TINYML_MODEL_PATH} (for TinyLCM compatibility)")
    
    model_size_mb = os.path.getsize(MODEL_TFLITE_PATH) / (1024 * 1024)
    print(f"TFLite model size: {model_size_mb:.2f} MB")
    
    # Also copy the labels file
    import shutil
    shutil.copy(LABELS_FILE_PATH, TINYML_LABELS_PATH)
    print(f"Labels copied to {TINYML_LABELS_PATH} for TinyLCM compatibility")
    
    return tflite_model

def test_tflite_compatibility():
    """Test that the TFLite model is compatible with TinyLCM preprocessors."""
    # Load the TFLite model from the TinyLCM-compatible path
    interpreter = tf.lite.Interpreter(model_path=TINYML_MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print("Testing TFLite model compatibility with TinyLCM:")
    print(f"Input shape: {input_details['shape']}")
    print(f"Input type: {input_details['dtype']}")
    print(f"Output shape: {output_details['shape']}")
    print(f"Output type: {output_details['dtype']}")
    
    # Check all output tensor details to find feature layers
    all_output_details = interpreter.get_output_details()
    print(f"Model has {len(all_output_details)} output tensors:")
    for i, output in enumerate(all_output_details):
        print(f"  Output {i}: shape={output['shape']}, name={output.get('name', 'unnamed')}")
    
    # Create a test image
    test_image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Test with TinyLCM preprocessor
    try:
        # Preprocess using TinyLCM utilities
        processed_image = prepare_input_tensor_quantized(test_image, input_details)
        
        # Set the input tensor
        interpreter.set_tensor(input_details['index'], processed_image)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output
        output = interpreter.get_tensor(output_details['index'])
        
        # Check output format and dimensionality
        print(f"Output tensor shape: {output.shape}")
        
        # Check if output is feature-rich (if it has a large number of dimensions)
        if len(output.shape) > 0 and (output.size > 100):
            print(f"✅ Feature layer has {output.size} dimensions - EXCELLENT for KNN classification!")
        else:
            print(f"⚠️ Feature layer has only {output.size} dimensions - may not be ideal for KNN")
            
        print(f"Output tensor: min={output.min():.5f}, max={output.max():.5f}")
        
        print("✅ TFLite model is compatible with TinyLCM preprocessors!")
        
    except Exception as e:
        print("❌ Error testing TFLite compatibility:")
        print(str(e))
        
    # Now explicitly try creating a KNN state with this model to verify
    try:
        # Import LightweightKNN to check state creation
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
        num_features = output.flatten().shape[0]  # Use the actual feature dimension
        
        # Create synthetic features and labels
        features = np.random.rand(num_samples, num_features)
        labels = ["negative", "lego", "stone", "tire"]
        timestamps = [time.time() - i*10 for i in range(num_samples)]
        
        # Fit the KNN
        knn.fit(features, labels, timestamps)
        
        # Get state
        state = knn.get_state()
        
        print(f"✅ Successfully created KNN state with {num_features}-dimensional features")
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
        rel_model_path = "../assets/model/model_object_transfer.tflite"
        rel_labels_path = "../assets/model/labels_object_transfer.txt"
        
        # Read existing config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Update model paths
        config["model"]["model_path"] = rel_model_path
        config["model"]["labels_path"] = rel_labels_path
        config["tinylcm"]["feature_extractor"]["model_path"] = rel_model_path
        
        # Write updated config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Updated config file: {config_file}")
        print(f"   - Set model_path to {rel_model_path}")
        print(f"   - Set labels_path to {rel_labels_path}")
    else:
        print(f"⚠️ Config file not found: {config_file}")

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
    
    print("\nModel training and conversion complete!")
    print(f"New model saved to {MODEL_TFLITE_PATH}")
    print(f"Labels file saved to {LABELS_FILE_PATH}")
    print("\nTo use the new model:")
    print("1. Run create_inital_knn_sc2.py to create a new KNN state")
    print("2. Run main_scenario2.py to test the new model")