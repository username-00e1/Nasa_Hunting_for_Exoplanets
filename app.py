from flask import Flask, render_template, request, jsonify
from model import ExoplanetModel
import os

app = Flask(__name__)
ml_model = ExoplanetModel()

# Load or train model on startup
if not ml_model.load_model():
    print("Training new model...")
    ml_model.train()
    print("Model trained successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        features = [
            float(data['orbital_period']),
            float(data['transit_duration']),
            float(data['planet_radius']),
            float(data['stellar_radius']),
            float(data['equilibrium_temp']),
            float(data['insolation_flux'])
        ]
        
        result = ml_model.predict(features)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train', methods=['POST'])
def train():
    try:
        metrics = ml_model.train()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
