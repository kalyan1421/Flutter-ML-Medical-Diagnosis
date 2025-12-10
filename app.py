# app.py

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from PIL import Image

# --- Configuration ---
# NOTE: Set the correct path to your model and labels
MODEL_PATH = '../Medical_AI_Diagnostic_System/models/keras/fetal_us_model.h5'
LABELS_PATH = '../Medical_AI_Diagnostic_System/models/tflite/fetal_us_labels.txt'

UPLOAD_FOLDER = 'static/uploads'
IMG_SIZE = 224 # Must match the size used during training (config.py)

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model and labels to load them ONCE
global model, class_names

def load_global_assets():
    """Loads the Keras model and class names into global memory."""
    print(" * Loading Keras model and labels...")
    
    # Load the Keras model. compile=False is often used to prevent issues in deployment.
    try:
        global model
        model = load_model(MODEL_PATH, compile=False)
        # Ensure model is ready for prediction (important for multi-threading/TensorFlow)
        # In newer TF versions, this line might be less critical but is good practice.
        # model._make_predict_function() 
        
        global class_names
        # Read labels, strip newline characters
        with open(LABELS_PATH, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        print(" * Model loaded successfully.")
        print(f" * Classes: {class_names}")

    except Exception as e:
        print(f"ERROR LOADING MODEL: {e}")
        model = None
        class_names = ["Model Load Error"]
        

def preprocess_image(image_path):
    """
    Preprocesses the uploaded image to match the model's expected input format.
    - Resizing to 224x224.
    - Converting to a NumPy array.
    - Normalizing pixel values (0 to 1.0).
    - Expanding dimensions for the model (1, 224, 224, 3).
    """
    # Load and resize the image
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    
    # Convert to array
    img_array = img_to_array(img)
    
    # Normalize (0-255 to 0.0-1.0)
    img_array /= 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    image_filename = None

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part', prediction_result=prediction_result)

        file = request.files['file']
        
        # If the user submits an empty part
        if file.filename == '':
            return render_template('index.html', error='No selected file', prediction_result=prediction_result)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            image_filename = filename

            if model is None:
                prediction_result = "Model is not loaded. Check server logs."
            else:
                try:
                    # Preprocess and predict
                    input_data = preprocess_image(file_path)
                    
                    # Make prediction
                    predictions = model.predict(input_data)
                    
                    # Get the predicted class index and confidence
                    predicted_index = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_index] * 100
                    
                    # Format the result
                    label = class_names[predicted_index]
                    prediction_result = f"{label} (Confidence: {confidence:.2f}%)"
                    
                except Exception as e:
                    prediction_result = f"Prediction failed: {e}"
                    print(f"Prediction Error: {e}")
                    
    # Render the HTML template, passing the result and image path
    return render_template(
        'index.html', 
        prediction_result=prediction_result,
        image_filename=image_filename
    )

# Run the app
if __name__ == '__main__':
    # Load the model only once when the server starts
    load_global_assets()
    # Debug=True is great for development but should be False in production
    app.run(debug=True)