#!/usr/bin/env python3
"""
TensorFlow Transfer Learning for Direct Classification (MobileNetV2)

This script trains a transfer learning model for direct object classification using
images from examples/assets/training_images. It creates a model that:
1. Uses MobileNetV2 as the base model (very efficient for Raspberry Pi Zero)
2. Outputs direct class predictions with softmax (no feature extraction)
3. Exports to TFLite format with quantization
4. Is compatible with scenario0_baseline_tflite

The model is saved in a format compatible with scenario0_baseline_tflite.
"""

import os
import pathlib
import subprocess
import sys

import numpy as np

# Check for required dependencies and install if missing
required_packages = ["Pillow", "scipy", "scikit-learn"]
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

# Install any missing packages
if missing_packages:
    print(f"Installing packages: {', '.join(missing_packages)}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
    print("All required packages installed successfully.")

# Import TensorFlow
import tensorflow as tf

print(f"Using TensorFlow version: {tf.__version__}")
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add the project root to the path
current_dir = pathlib.Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

# Configuration
IMAGE_SIZE = (224, 224)  # MobileNetV2 default input size
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2
CLASS_MODE = 'categorical'  # One-hot encoded labels

# Directories
IMAGE_DIR = os.path.join(root_dir, "examples/assets/training_images")
OUTPUT_DIR = os.path.join(root_dir, "examples/scenario0_baseline_tflite/model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file paths
MODEL_KERAS_PATH = os.path.join(OUTPUT_DIR, "model_classification.h5")
MODEL_TFLITE_PATH = os.path.join(OUTPUT_DIR, "model_classification.tflite")
LABELS_FILE_PATH = os.path.join(OUTPUT_DIR, "labels_classification.txt")


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
    
    # Save labels in the correct format
    labels = sorted(train_generator.class_indices.keys())
    with open(LABELS_FILE_PATH, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    
    print(f"Saved class labels to {LABELS_FILE_PATH}")
    print(f"Training generator found {train_generator.samples} samples")
    print(f"Validation generator found {val_generator.samples} samples")
    
    return train_generator, val_generator, len(class_dirs)


def create_classification_model(num_classes):
    """Create a transfer learning model with MobileNetV2 for direct classification."""
    print(f"Creating classification model with MobileNetV2...")
    
    # Create the MobileNetV2 base model without top layers
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom classification layers
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    print(f"Model creation complete:")
    print(f"- Input shape: {model.input_shape}")
    print(f"- Output shape: {model.output_shape}")
    print(f"- Total parameters: {model.count_params():,}")
    
    return model


def train_model(model, train_generator, val_generator):
    """Train the classification model."""
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
            patience=15,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-6
        )
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = val_generator.samples // BATCH_SIZE
    
    print("Starting model training...")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=max(1, validation_steps)
    )
    
    return history


def fine_tune_model(model, train_generator, val_generator):
    """Fine-tune the model by unfreezing some layers."""
    # Unfreeze the last few layers of the base model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # This is the base model
            # Unfreeze the last 20 layers
            for sublayer in layer.layers[-20:]:
                sublayer.trainable = True
            break
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
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
        epochs=EPOCHS // 2,
        validation_data=val_generator,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=max(1, validation_steps)
    )
    
    return history


def convert_to_tflite(model, val_generator, quantize=True):
    """Convert the model to TFLite format with optional quantization."""
    print("\n==== Converting Model to TFLite ====")
    
    # Save the Keras model first
    model.save(MODEL_KERAS_PATH)
    print(f"Saved Keras model to {MODEL_KERAS_PATH}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("Applying int8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configure int8 quantization
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        
        # Set the inference input/output type
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32  # Keep output as float32 for softmax
        
        # Representative dataset for calibration
        def representative_data_gen():
            num_calibration_examples = 100
            print(f"Generating calibration data from {num_calibration_examples} examples...")
            count = 0
            
            for input_value, _ in val_generator:
                if count >= num_calibration_examples:
                    break
                    
                for i in range(min(BATCH_SIZE, input_value.shape[0])):
                    if count >= num_calibration_examples:
                        break
                    
                    yield [input_value[i:i+1]]
                    count += 1
                    
                    if count % 10 == 0:
                        print(f"  Processed {count}/{num_calibration_examples} calibration examples")
        
        converter.representative_dataset = representative_data_gen
    
    print("Converting model to TFLite format...")
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(MODEL_TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    model_size_mb = os.path.getsize(MODEL_TFLITE_PATH) / (1024 * 1024)
    print(f"✅ TFLite model saved to: {MODEL_TFLITE_PATH}")
    print(f"   Model size: {model_size_mb:.2f} MB")
    
    return tflite_model


def test_tflite_model():
    """Test the TFLite model to ensure it outputs correct predictions."""
    print("\n==== Testing TFLite Model ====")
    
    # Load the TFLite model
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
    
    # Create a test image
    test_image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    
    # Prepare input
    input_tensor = np.expand_dims(test_image, axis=0)
    if input_details['dtype'] == np.float32:
        input_tensor = input_tensor.astype(np.float32) / 255.0
    
    # Set input tensor
    interpreter.set_tensor(input_details['index'], input_tensor)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details['index'])
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output sum: {output.sum():.4f} (should be close to 1.0 for softmax)")
    print(f"Output sample: {output[0][:5]}...")
    
    # Load labels
    with open(LABELS_FILE_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Get prediction
    predicted_idx = np.argmax(output[0])
    confidence = output[0][predicted_idx]
    predicted_class = labels[predicted_idx]
    
    print(f"\nTest prediction:")
    print(f"Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\n✅ TFLite model test complete!")


if __name__ == "__main__":
    print("Starting transfer learning for Direct Classification Model")
    print(f"Using images from: {IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Create data generators
    train_generator, val_generator, num_classes = create_data_generators()
    
    # Create the classification model
    model = create_classification_model(num_classes)
    print("\nModel architecture:")
    model.summary()
    
    # Train the model
    history = train_model(model, train_generator, val_generator)
    
    # Fine-tune the model
    history = fine_tune_model(model, train_generator, val_generator)
    
    # Convert to TFLite format
    tflite_model = convert_to_tflite(model, val_generator, quantize=True)
    
    # Test the TFLite model
    test_tflite_model()
    
    print("\n===============================================")
    print("✅ CLASSIFICATION MODEL TRAINING COMPLETE")
    print("===============================================")
    print(f"Model saved to: {MODEL_TFLITE_PATH}")
    print(f"Labels saved to: {LABELS_FILE_PATH}")
    
    print("\nFeatures:")
    print("✓ Optimized for Raspberry Pi Zero 2W")
    print("✓ Direct classification output with softmax")
    print("✓ Int8 quantized for efficient inference")
    print("✓ Compatible with scenario0_baseline_tflite")
    
    print("\nTo use the model:")
    print("1. Update the config_scenario0.json to use the new model paths")
    print("2. Run: python main_scenario0.py")