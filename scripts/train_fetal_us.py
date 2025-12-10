# scripts/train_fetal_us.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import config
import os
from sklearn.utils import class_weight
import numpy as np

# --- PATH FIX: Ensure the output directories exist ---
os.makedirs(os.path.dirname(config.FETAL_US_KERAS_MODEL), exist_ok=True)
os.makedirs(os.path.dirname(config.FETAL_US_LABELS), exist_ok=True) 

# --- 1. Data Generators (Data Augmentation) ---
print("Setting up Data Generators...")
# Keep the same aggressive augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=30, width_shift_range=0.3,  
    height_shift_range=0.3, zoom_range=0.3, horizontal_flip=True,   
    vertical_flip=True, fill_mode='nearest'
)
validation_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    f'{config.FETAL_US_DATA_PATH}/train',
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical'
)
validation_generator = validation_test_datagen.flow_from_directory(
    f'{config.FETAL_US_DATA_PATH}/validation',
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical'
)
test_generator = validation_test_datagen.flow_from_directory(
    f'{config.FETAL_US_DATA_PATH}/test',
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- CRITICAL FIX: Calculate and Apply Class Weights ---
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print("\nCalculated Class Weights (Higher weight means rarer class):")
print(class_weights_dict)

# Store the class labels
class_labels = list(train_generator.class_indices.keys())
with open(config.FETAL_US_LABELS, 'w') as f:
    for label in class_labels:
        f.write(f"{label}\n")
print(f"Labels saved to: {config.FETAL_US_LABELS}")


# --- 2. Model Building (Optimized Head) ---
print("\nBuilding MobileNetV2 Model...")
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
)
base_model.trainable = False 

# Enhanced Classification Head for stability and generalization
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(), # Added for stability
    Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.005)), # Stronger L2
    Dropout(config.DROPOUT_RATE), 
    Dense(config.FETAL_US_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2(0.005)) # L2 on final layer
])

# Define Callbacks
callbacks = [
    ModelCheckpoint(
        filepath=config.FETAL_US_KERAS_MODEL,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy', 
        patience=10, # Increased patience to give weighted loss time to converge
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
]

# --- 3. Stage 1: Feature Extraction (WITH CLASS WEIGHTS) ---
print(f"\n--- Stage 1: Feature Extraction ({config.INITIAL_EPOCHS} Epochs) ---")
model.compile(
    optimizer=Adam(learning_rate=config.INITIAL_LEARNING_RATE), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.fit(
    train_generator,
    epochs=config.INITIAL_EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights_dict # CRITICAL: Applying Class Weights
)


# --- 4. Stage 2: Fine-Tuning (WITH CLASS WEIGHTS) ---
print(f"\n--- Stage 2: Fine-Tuning ({config.FINE_TUNE_EPOCHS} Epochs) ---")

base_model.trainable = True
# Only unfreeze the *last* 25% of the layers (index calculated by 0.75)
fine_tune_at = int(len(base_model.layers) * 0.75) 
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False 

model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE_FT), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

fine_tune_callbacks = [
    ModelCheckpoint(
        filepath=config.FETAL_US_KERAS_MODEL, 
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy', 
        patience=15, # Further increased patience for fine-tuning
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, # Increased patience
        min_lr=1e-7,
        verbose=1
    )
]

model.fit(
    train_generator,
    epochs=config.INITIAL_EPOCHS + config.FINE_TUNE_EPOCHS, 
    initial_epoch=model.history.epoch[-1],
    validation_data=validation_generator,
    callbacks=fine_tune_callbacks,
    class_weight=class_weights_dict # CRITICAL: Applying Class Weights
)

# --- 5. Final Evaluation ---
print("\n--- Final Evaluation on Test Data (Using Best Saved Weights) ---")
best_model = tf.keras.models.load_model(config.FETAL_US_KERAS_MODEL)

from sklearn.metrics import classification_report
predictions = best_model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

test_loss, test_acc = best_model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")

print("\n--- Detailed Classification Report ---")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))