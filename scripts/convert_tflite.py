# scripts/convert_tflite.py

import tensorflow as tf
import config
import os

def convert_model(keras_path, tflite_path):
    """Loads Keras model, optimizes, and saves as TFLite."""
    if not os.path.exists(keras_path):
        print(f"Error: Keras model not found at {keras_path}. Skipping conversion.")
        return

    print(f"Converting {os.path.basename(keras_path)}...")
    
    # Load the Keras model
    model = tf.keras.models.load_model(keras_path)

    # Instantiate the TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimization: Default quantization for size reduction and speed
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 

    try:
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Successfully converted and saved to: {tflite_path}")

    except Exception as e:
        print(f"Conversion failed for {keras_path}: {e}")

if __name__ == '__main__':
    # Ensure TFLite output directory exists
    os.makedirs(os.path.dirname(config.LUNG_CANCER_TFLITE_MODEL), exist_ok=True)
    
    # Convert Lung Cancer Model
    convert_model(config.LUNG_CANCER_KERAS_MODEL, config.LUNG_CANCER_TFLITE_MODEL)

    # Convert Fetal Ultrasound Model
    convert_model(config.FETAL_US_KERAS_MODEL, config.FETAL_US_TFLITE_MODEL)

    # Convert Brain Hemorrhage Model
    convert_model(config.BRAIN_HEM_KERAS_MODEL, config.BRAIN_HEM_TFLITE_MODEL)
    
    print("\nAll conversions attempted. Check the '../models/tflite/' folder.")