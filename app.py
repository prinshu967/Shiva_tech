from flask import Flask, request, jsonify, render_template, redirect

from PIL import Image
import numpy as np
import tensorflow as tf
import math

import os

app = Flask(__name__)

# Load your trained model (make sure the path is correct)
model = tf.keras.models.load_model('New2Freshness50.h5')  # Load the Keras model

# Define the class labels (update these as per your model)
class_labels = ['Bad Apple', 'Bad Banana', 'Bad Bellpepper', 'Bad Cucumber', 'Bad Grapes',
                'Bad Indian Green Chile', 'Bad Mango', 'Bad Orange', 'Bad Potato', 'Bad Tomato',
                'Fresh Apple', 'Fresh Banana', 'Fresh Bellpepper', 'Fresh Cucumber', 'Fresh Grapes',
                'Fresh Indian Green Chile', 'Fresh Mango', 'Fresh Orange', 'Fresh Potato', 'Fresh Tomato',
                'Moderate Apple', 'Moderate Banana', 'Moderate Bellpepper', 'Moderate Cucumber',
                'Moderate Grapes', 'Moderate Indian Green Chile', 'Moderate Mango', 'Moderate Orange',
                'Moderate Potato', 'Moderate Tomato']

# Arrhenius equation-related factors
PreExpontialFactor = {
    'Fresh Apple': 645092.1348576404, 'Fresh Banana': 15370.169119416576, 'Fresh Bellpepper': 23330.754782432436, 
    'Fresh Cucumber': 277.94296916110966, 'Fresh Grapes': 23330.754782432436, 'Fresh Indian Green Chile': 23330.754782432436, 
    'Fresh Mango': 277.94296916110966, 'Fresh Orange': 5.995771324920545, 'Fresh Potato': 295.3274908449173, 
    'Fresh Tomato': 15370.169119416576, 'Moderate Apple': 15370.169119416576, 'Moderate Banana': 11139.41897966447, 
    'Moderate Bell Pepper': 113.07685698928096, 'Moderate Cucumber': 3494.564827089312, 'Moderate Grapes': 440004.8432500969, 
    'Moderate Indian Green Chile': 113.07685698928096, 'Moderate Mango': 208.10538328031532, 'Moderate Orange': 277.94296916110966, 
    'Moderate Potato': 1747.282413544656, 'Moderate Tomato': 440004.8432500969
}

AcivationEnergy = {
    'Fresh Apple': 38867.22294049396, 'Fresh Banana': 27885.777266933837, 'Fresh Bellpepper': 29754.351047873723, 
    'Fresh Cucumber': 18772.9053743136, 'Fresh Grapes': 29754.351047873723, 'Fresh Indian Green Chile': 29754.351047873723, 
    'Fresh Mango': 18772.9053743136, 'Fresh Orange': 10981.445673560118, 'Fresh Potato': 20641.479155253488, 
    'Fresh Tomato': 27885.777266933837, 'Moderate Apple': 27885.777266933837, 'Moderate Banana': 24816.431036967108, 
    'Moderate Bell Pepper': 15156.397555273734, 'Moderate Cucumber': 22947.857256027222, 'Moderate Grapes': 33929.302929587335, 
    'Moderate Indian Green Chile': 15156.397555273734, 'Moderate Mango': 16191.564734321395, 'Moderate Orange': 18772.9053743136, 
    'Moderate Potato': 22947.857256027222, 'Moderate Tomato': 33929.302929587335
}

# Arrhenius equation: calculates the rate constant 'k' and shelf life
def calculate_rate_constant(Ea, A, T):
    R = 8.314  # Universal gas constant in J/(mol*K)
    k = A * math.exp(-Ea / (R * T))
    return k

def predict_shelf_life(Ea, A, T):
    k = calculate_rate_constant(Ea, A, T)
    shelf_life = 1 / k
    return shelf_life

# Function to load an image and make predictions
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return None
    
    try:
        # Load and preprocess image
        img = Image.open(image_path)
        img_resized = img.resize((224, 224))  # Resize to match VGG16's input shape
        img_array = np.array(img_resized) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        return predicted_class
    except Exception as e:
        print(f"Error in predicting image: {e}")
        return None

# Flask app setup
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file:
        # Save uploaded file
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)
        
        # Predict fruit freshness
        predicted_class = predict_image(filepath)
        
        if predicted_class:
            # If fruit is bad
            if "Bad" in predicted_class:
                return render_template('index.html', predicted_class=predicted_class, shelf_life=0)
            else:
                # Get temperature input from user and calculate shelf life
                temperature = float(request.form['temperature']) + 273.15  # Convert Celsius to Kelvin
                Ea = AcivationEnergy[predicted_class]
                A = PreExpontialFactor[predicted_class]
                shelf_life = predict_shelf_life(Ea, A, temperature)
                
                return render_template('index.html', predicted_class=predicted_class, shelf_life=shelf_life)
        else:
            return render_template('index.html', error="Prediction failed.")
    return render_template('index.html', error="No file selected.")

if __name__ == '__main__':
    app.run(debug=True)
