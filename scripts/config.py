# scripts/config.py

import os

# --- Global ML Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
# Initial Epochs for feature extraction (frozen base model)
INITIAL_EPOCHS = 10 
# Fine-tuning Epochs (unfrozen base model)
FINE_TUNE_EPOCHS = 30 
LEARNING_RATE_FT = 1e-5 # Very low learning rate for fine-tuning
INITIAL_LEARNING_RATE = 0.0001
OPTIMIZATION = 'Adam'
DROPOUT_RATE = 0.4 # Regularization to prevent overfitting

# --- Base Paths ---
BASE_DATA_DIR = '../datasets'
BASE_MODELS_DIR = '../models'

# --- Fetal Ultrasound Paths ---
FETAL_US_CLASSES = 3  # Normal, Benign, Malignant
FETAL_US_DATA_PATH = f'{BASE_DATA_DIR}/fetal_us_data'
FETAL_US_KERAS_MODEL = f'{BASE_MODELS_DIR}/keras/fetal_us_model.h5'
FETAL_US_TFLITE_MODEL = f'{BASE_MODELS_DIR}/tflite/fetal_us_model.tflite'
FETAL_US_LABELS = f'{BASE_MODELS_DIR}/tflite/fetal_us_labels.txt'

# --- Define the Fine-Tune Point (Example: Unfreeze the last 50% of the base model layers)
# This controls which layers will be trained during the fine-tuning stage.
# We will calculate the exact layer index in the training script.
FINE_TUNE_PERCENTAGE = 0.5