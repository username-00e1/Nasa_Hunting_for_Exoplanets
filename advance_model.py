# advanced_models.py
# Advanced ML models including Neural Networks and Ensemble methods

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# ============================================
# 1. DEEP LEARNING MODEL (Neural Network)
# ============================================

class ExoplanetNeuralNetwork:
    """Deep learning model for exoplanet classification"""
    
    def __init__(self, input_dim, num_classes=3):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, architecture='deep'):
        """Build neural network architecture"""
        
        if architecture == 'deep':
            # Deep neural network with dropout
            self.model = models.Sequential([
                layers.Dense(256, activation='relu', input_shape=(self.input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        elif architecture == 'cnn':
            # 1D CNN for time-series light curve data
            self.model = models.Sequential([
                layers.Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
                
                layers.Conv1D(64, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2),
                layers.Dropout(0.3),
                
                layers.Conv1D(128, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2),
                layers.Dropout(0.3),
                
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the neural network"""
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred, y_pred_proba
    
    def plot_training_history(self):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath='exoplanet_nn.h5'):
        """Save neural network model"""
        self.model.save(filepath)
        print(f"Neural network saved to {filepath}")
    
    def load_model(self, filepath='exoplanet_nn.h5'):
        """Load neural network model"""
        self.model = keras.models.load_model(filepath)
        print(f"Neural network loaded from {filepath}")


# ============================================
# 2. GRADIENT BOOSTING MODELS
# ============================================

class XGBoostClassifier:
    """XGBoost model for exoplanet classification"""
    
    def __init__(self):
        self.model = None
        
    def build_model(self, class_weights=None):
        """Build XGBoost model"""
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50
        )
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities
    
    def get_feature_importance(self):
        """Get feature importance"""
        return self.model.feature_importances_


class LightGBMClassifier:
    """LightGBM model for exoplanet classification"""
    
    def __init__(self):
        self.model = None
        
    def build_model(self):
        """Build LightGBM model"""
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            objective='multiclass',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities


# ============================================
# 3. ENSEMBLE MODELS
# ============================================

class EnsembleClassifier:
    """Ensemble of multiple models for better predictions"""
    
    def __init__(self, base_models=None):
        self.base_models = base_models or []
        self.ensemble_model = None
        
    def build_voting_ensemble(self, models_dict):
        """
        Build voting ensemble
        models_dict: {'name': model_instance}
        """
        estimators = [(name, model) for name, model in models_dict.items()]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability predictions
            n_jobs=-1
        )
        
        return self.ensemble_model
    
    def build_stacking_ensemble(self, base_models, meta_model=None):
        """
        Build stacking ensemble
        base_models: list of (name, model) tuples
        meta_model: model for final prediction (default: LogisticRegression)
        """
        if meta_model is None:
            meta_model = LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                random_state=42
            )
        
        self.ensemble_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        return self.ensemble_model
    
    def train(self, X_train, y_train):
        """Train ensemble model"""
        print("Training ensemble model...")
        self.ensemble_model.fit(X_train, y_train)
        print("Ensemble training completed!")
        return self.ensemble_model
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.ensemble_model.predict(X)
        probabilities = self.ensemble_model.predict_proba(X)
        return predictions, probabilities


# ============================================
# 4. LIGHT CURVE ANALYZER
# ============================================

class LightCurveAnalyzer:
    """
    Analyze transit light curves for exoplanet detection
    Works with time-series photometric data
    """
    
    def __init__(self):
        self.model = None
        
    def preprocess_light_curve(self, time, flux):
        """
        Preprocess light curve data
        - Normalize flux
        - Remove outliers
        - Detrend
        """
        # Normalize flux
        flux_normalized = (flux - np.median(flux)) / np.median(flux)
        
        # Remove outliers using sigma clipping
        sigma = np.std(flux_normalized)
        mask = np.abs(flux_normalized) < 3 * sigma
        time_clean = time[mask]
        flux_clean = flux_normalized[mask]
        
        # Simple detrending (linear)
        coeffs = np.polyfit(time_clean, flux_clean, 1)
        trend = np.polyval(coeffs, time_clean)
        flux_detrended = flux_clean - trend
        
        return time_clean, flux_detrended
    
    def detect_transit(self, time, flux, threshold=-0.01):
        """
        Detect transit events in light curve
        Returns: transit times and depths
        """
        # Find points below threshold
        transit_mask = flux < threshold
        
        if not np.any(transit_mask):
            return None, None
        
        # Find transit windows
        transit_starts = np.where(np.diff(transit_mask.astype(int)) == 1)[0]
        transit_ends = np.where(np.diff(transit_mask.astype(int)) == -1)[0]
        
        transit_times = []
        transit_depths = []
        
        for start, end in zip(transit_starts, transit_ends):
            transit_window = flux[start:end+1]
            transit_times.append(time[start:end+1].mean())
            transit_depths.append(transit_window.min())
        
        return np.array(transit_times), np.array(transit_depths)
    
    def calculate_transit_features(self, time, flux):
        """
        Calculate features from light curve
        """
        features = {}
        
        # Transit detection
        transit_times, transit_depths = self.detect_transit(time, flux)
        
        if transit_times is not None and len(transit_times) > 1:
            # Orbital period (time between transits)
            features['period'] = np.median(np.diff(transit_times))
            
            # Average transit depth
            features['depth'] = np.abs(np.median(transit_depths))
            
            # Transit duration (rough estimate)
            features['duration'] = features['period'] * 0.1  # Simplified
            
            # Signal-to-noise ratio
            noise = np.std(flux)
            features['snr'] = features['depth'] / noise if noise > 0 else 0
        else:
            # No clear transits detected
            features = {
                'period': 0,
                'depth': 0,
                'duration': 0,
                'snr': 0
            }
        
        return features
    
    def build_cnn_model(self, input_length):
        """Build 1D CNN for raw light curve classification"""
        model = models.Sequential([
            layers.Input(shape=(input_length, 1)),
            
            # First convolutional block
            layers.Conv1D(16, 5, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # Second convolutional block
            layers.Conv1D(32, 5, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # Third convolutional block
            layers.Conv1D(64, 5, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model


# ============================================
# 5. MODEL COMPARISON & SELECTION
# ============================================

class ModelComparison:
    """Compare multiple models and select the best one"""
    
    def __init__(self):
        self.results = {}
        
    def add_model_results(self, name, y_true, y_pred, y_prob):
        """Add results for a model"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        from sklearn.preprocessing import label_binarize
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Multi-class AUC
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        try:
            auc = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
        except:
            auc = 0.0
        
        self.results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
    
    def get_comparison_table(self):
        """Get comparison table as DataFrame"""
        import pandas as pd
        df = pd.DataFrame(self.results).T
        df = df.round(4)
        return df.sort_values('f1_score', ascending=False)
    
    def plot_comparison(self):
        """Plot model comparison"""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        df = self.get_comparison_table()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df))
        width = 0.15
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (i - 2)
            ax.bar(x + offset, df[metric], width, label=metric.replace('_', ' ').title(), 
                   color=color, alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig
    
    def get_best_model(self, metric='f1_score'):
        """Get best model name based on metric"""
        df = self.get_comparison_table()
        return df[metric].idxmax()


# ============================================
# Usage Example
# ============================================

if __name__ == '__main__':
    print("Advanced ML Models Module Loaded")
    print("Available models:")
    print("  - ExoplanetNeuralNetwork (Deep Learning)")
    print("  - XGBoostClassifier")
    print("  - LightGBMClassifier")
    print("  - EnsembleClassifier")
    print("  - LightCurveAnalyzer (for time-series data)")
