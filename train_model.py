"""
=============================================================
  LUNG CANCER DETECTION - CNN MODEL TRAINING SCRIPT
  Final Year Project
=============================================================
  Dataset: IQ-OTH/NCCD Lung Cancer Dataset
  Classes: Malignant, Benign, Normal
  Model: Custom CNN + Transfer Learning (MobileNetV2)
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
#  1. CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE    = (224, 224)   # Input image size
BATCH_SIZE  = 32
EPOCHS      = 30
NUM_CLASSES = 3
LR          = 0.0001       # Learning rate

# ⚠️  UPDATE THESE PATHS to match your local machine
TRAIN_DIR = "dataset/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset"
TEST_DIR  = "dataset/Test cases"
MODEL_SAVE_PATH = "model/lung_cancer_cnn.h5"

os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("✅ TensorFlow version:", tf.__version__)
print("✅ GPU available:", tf.config.list_physical_devices('GPU'))

# ─────────────────────────────────────────────
#  2. DATA LOADING & AUGMENTATION
# ─────────────────────────────────────────────
# Data augmentation helps the model generalise better
# by artificially creating more training samples

train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values 0-1
    rotation_range=15,         # Randomly rotate images
    width_shift_range=0.1,     # Randomly shift horizontally
    height_shift_range=0.1,    # Randomly shift vertically
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,      # Flip images horizontally
    fill_mode='nearest',
    validation_split=0.2       # 20% data used for validation
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("\n📊 Class mapping:", train_generator.class_indices)
print("📊 Training samples:", train_generator.samples)
print("📊 Validation samples:", val_generator.samples)
print("📊 Test samples:", test_generator.samples)

CLASS_NAMES = list(train_generator.class_indices.keys())

# ─────────────────────────────────────────────
#  3. HANDLE CLASS IMBALANCE
# ─────────────────────────────────────────────
# Our dataset has more Malignant than Benign images
# Class weights fix this imbalance

labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights_array))
print("\n⚖️  Class weights:", class_weight_dict)

# ─────────────────────────────────────────────
#  4. BUILD THE CNN MODEL (Transfer Learning)
# ─────────────────────────────────────────────
# We use MobileNetV2 pre-trained on ImageNet
# This is called Transfer Learning - using knowledge
# from a large dataset to help with our smaller dataset

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,          # Remove original classification head
    weights='imagenet'          # Use pre-trained weights
)

# Freeze base model layers (don't retrain them initially)
base_model.trainable = False

# Build our custom classifier on top
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)        # Reduce feature maps to 1D
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)   # Fully connected layer
x = Dropout(0.5)(x)                    # Prevent overfitting
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)  # 3-class output

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────
#  5. CALLBACKS (Auto-save & Early Stop)
# ─────────────────────────────────────────────
callbacks = [
    # Save best model automatically
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Stop training if no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ─────────────────────────────────────────────
#  6. TRAIN THE MODEL - PHASE 1
# ─────────────────────────────────────────────
print("\n🚀 Phase 1: Training with frozen base model...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
#  7. FINE-TUNING - PHASE 2
# ─────────────────────────────────────────────
# Unfreeze top layers of base model for fine-tuning
print("\n🔧 Phase 2: Fine-tuning top layers...")

base_model.trainable = True
# Only fine-tune top 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LR / 10),   # Much lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
#  8. EVALUATE ON TEST SET
# ─────────────────────────────────────────────
print("\n📊 Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# Predictions
test_generator.reset()
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
#  9. PLOT RESULTS
# ─────────────────────────────────────────────
def plot_training_history(history, history_fine=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if history_fine:
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']
        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='red')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='green')
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: plots/training_history.png")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: plots/confusion_matrix.png")


plot_training_history(history, history_fine)
plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)

print("\n🎉 Training Complete!")
print(f"📁 Model saved at: {MODEL_SAVE_PATH}")
