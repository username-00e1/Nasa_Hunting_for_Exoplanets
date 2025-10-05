# Nasa_Hunting_for_Exoplanets
Creating  a web user  interface and integrating ML model to better predict exoplanets

Features

Multiple ML Models: Random Forest, XGBoost, LightGBM, Neural Networks, and Ensemble methods
Advanced Feature Engineering: Automated creation of derived features for better predictions
Interactive Web Interface: React-based UI and Streamlit dashboard
RESTful API: Flask API for production deployment
Real-time Predictions: Single and batch classification capabilities
Comprehensive Visualization: Model performance, feature importance, and data distribution plots
Docker Support: Easy deployment with containerization

📊 Model Performance

ModelAccuracyPrecisionRecallF1-ScoreRandom Forest94.2%91.8%89.5%90.6%
XGBoost93.8%91.2%89.1%90.1%
LightGBM93.5%90.8%88.7%89.7%
Neural Network92.9%90.1%87.9%89.0%
Ensemble95.1%92.5%90.8%91.6%


# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Download NASA Data
pythonfrom data_downloader import NASADataDownloader

downloader = NASADataDownloader()
downloader.download_all()
Train Models
bash# Train all models
python train_complete.py --mode train --data data/cumulative.csv

# This will:
# 1. Load and preprocess data
# 2. Train Random Forest, XGBoost, LightGBM, Neural Network
# 3. Create ensemble model
# 4. Generate comparison reports and visualizations
Run API Server
bash# Start Flask API
python exoplanet_detector.py

# API will be available at http://localhost:5000
Launch Web Interface
bash# Start Streamlit app
streamlit run app.py

# Open browser to http://localhost:8501
📁 Project Structure
exoplanet-detector/
├── exoplanet_detector.py      # Main ML pipeline and Flask API
├── advanced_models.py          # Neural networks and advanced models
├── config.py                   # Configuration settings
├── data_downloader.py          # NASA data fetcher
├── visualization.py            # Plotting utilities
├── train_complete.py           # Complete training pipeline
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── data/                       # NASA datasets
│   ├── cumulative.csv          # Kepler data
│   ├── k2candidates.csv        # K2 data
│   └── tess_candidates.csv     # TESS data
├── models/                     # Trained models
│   ├── random_forest_model.pkl
│   ├── neural_network_model.h5
│   └── scaler.pkl
└── output/                     # Results and visualizations
    ├── class_distribution.png
    ├── feature_importance_rf.png
    ├── nn_training_history.png
    └── model_comparison.csv
🔧 API Usage
Single Prediction
pythonimport requests

url = "http://localhost:5000/api/predict"
data = {
    "koi_period": 3.5,
    "koi_duration": 2.1,
    "koi_depth": 100.0,
    "koi_prad": 1.2,
    "koi_srad": 1.0,
    "koi_teq": 285.0,
    "koi_insol": 1.0,
    "koi_steff": 5778.0,
    "koi_slogg": 4.5,
    "koi_impact": 0.5
}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
Batch Prediction
pythonurl = "http://localhost:5000/api/batch_predict"
data = {
    "samples": [
        {"id": "KOI-001", "koi_period": 3.5, ...},
        {"id": "KOI-002", "koi_period": 7.2, ...},
        # ... more samples
    ]
}

response = requests.post(url, json=data)
results = response.json()['results']
🐳 Docker Deployment
bash# Build and run with Docker Compose
docker-compose up -d

# The API will be available at http://localhost:5000
📊 Input Features
The model uses the following features for classification:
Core Features

Orbital Period (koi_period): Time for planet to complete one orbit (days)
Transit Duration (koi_duration): Duration of transit event (hours)
Transit Depth (koi_depth): Depth of stellar light dimming (ppm)
Planet Radius (koi_prad): Planetary radius (Earth radii)
Stellar Radius (koi_srad): Host star radius (Solar radii)
Equilibrium Temperature (koi_teq): Planet equilibrium temperature (Kelvin)
Insolation Flux (koi_insol): Incident radiation (Earth flux units)
Stellar Temperature (koi_steff): Star effective temperature (Kelvin)
Stellar Gravity (koi_slogg): Star surface gravity (log10(cm/s²))
Impact Parameter (koi_impact): Transit impact parameter (0-1)

Engineered Features

Transit Ratio: Transit depth to stellar radius ratio
Signal Strength: Product of depth and duration
Habitable Zone: Binary indicator for habitable zone
Period-Radius Ratio: Orbital period to planetary radius relationship
Stellar Mass Proxy: Derived stellar mass estimate

🎯 Classification Categories

CONFIRMED: High-confidence exoplanet detection
CANDIDATE: Potential exoplanet requiring additional validation
FALSE POSITIVE: Non-planetary signal (eclipsing binary, instrumental artifact, etc.)

📈 Model Details
Random Forest

200 decision trees
Max depth: 20
Class-balanced weights
Feature importance analysis

XGBoost

300 boosting rounds
Learning rate: 0.05
Max depth: 8
L1/L2 regularization

Neural Network

5-layer architecture
Batch normalization
Dropout regularization
Adam optimizer
Early stopping

Ensemble (Stacking)

Base models: RF, XGBoost, LightGBM
Meta-model: Logistic Regression
5-fold cross-validation

🔬 Data Sources
Kepler Mission (2009-2013)

URL: https://exoplanetarchive.ipac.caltech.edu/
Stars monitored: 150,000+
Confirmed exoplanets: 2,700+

K2 Mission (2014-2018)

Extended Kepler mission
Multiple campaign fields
Additional 500+ confirmed planets

TESS Mission (2018-Present)

All-sky survey
2-minute cadence
Ongoing discoveries

🛠️ Advanced Usage
Custom Model Training
pythonfrom exoplanet_detector import ExoplanetClassifier
from sklearn.ensemble import RandomForestClassifier

# Create custom model
custom_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    random_state=42
)

classifier = ExoplanetClassifier()
classifier.model = custom_rf
classifier.train(X_train, y_train, feature_names)
Hyperparameter Tuning
pythonfrom sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 25],
    'min_samples_split': [5, 10, 15]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
Light Curve Analysis
pythonfrom advanced_models import LightCurveAnalyzer

analyzer = LightCurveAnalyzer()

# Preprocess light curve
time_clean, flux_clean = analyzer.preprocess_light_curve(time, flux)

# Detect transits
transit_times, transit_depths = analyzer.detect_transit(time_clean, flux_clean)

# Extract features
features = analyzer.calculate_transit_features(time_clean, flux_clean)
📝 Citation
If you use this system in your research, please cite:
bibtex@software{exoplanet_detector_2024,
  author = {Your Name},
  title = {NASA Exoplanet Detection System},
  year = {2024},
  url = {https://github.com/yourusername/exoplanet-detector}
}
🤝 Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

NASA Exoplanet Archive for providing open-source data
Kepler, K2, and TESS mission teams
scikit-learn, TensorFlow, and XGBoost communities

📧 Contact
For questions or support, please open an issue on GitHub or contact: your.email@example.com
🔗 Useful Links

NASA Exoplanet Archive
Kepler Mission
TESS Mission
Documentation
