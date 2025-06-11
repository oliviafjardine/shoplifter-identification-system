import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pickle
import os

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_history = []
        self.model_path = "anomaly_model.pkl"
        self.scaler_path = "anomaly_scaler.pkl"
        
        # Load existing model if available
        self._load_model()
    
    def extract_features(self, behavior_data: Dict, track_data: Dict) -> np.ndarray:
        """
        Extract features from behavior and tracking data for anomaly detection
        """
        features = []
        
        # Basic behavior features
        features.append(behavior_data.get('suspicious_score', 0))
        
        # Movement features
        positions = track_data.get('positions', [])
        if len(positions) >= 2:
            # Calculate movement statistics
            distances = []
            speeds = []
            
            for i in range(1, len(positions)):
                curr_pos = positions[i]
                prev_pos = positions[i-1]
                
                distance = np.sqrt((curr_pos['x'] - prev_pos['x'])**2 + 
                                 (curr_pos['y'] - prev_pos['y'])**2)
                distances.append(distance)
                
                time_diff = (curr_pos['timestamp'] - prev_pos['timestamp']).total_seconds()
                if time_diff > 0:
                    speed = distance / time_diff
                    speeds.append(speed)
            
            # Movement statistics features
            features.extend([
                np.mean(distances) if distances else 0,
                np.std(distances) if len(distances) > 1 else 0,
                np.mean(speeds) if speeds else 0,
                np.std(speeds) if len(speeds) > 1 else 0,
                len(distances),  # Number of movements
            ])
            
            # Area coverage features
            x_coords = [pos['x'] for pos in positions]
            y_coords = [pos['y'] for pos in positions]
            features.extend([
                max(x_coords) - min(x_coords),  # X range
                max(y_coords) - min(y_coords),  # Y range
                np.std(x_coords),  # X spread
                np.std(y_coords),  # Y spread
            ])
        else:
            # Default values when insufficient position data
            features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # Time-based features
        first_seen = track_data.get('first_seen')
        last_seen = track_data.get('last_seen')
        if first_seen and last_seen:
            duration = (last_seen - first_seen).total_seconds()
            features.append(duration)
            
            # Time of day features (normalized)
            hour = first_seen.hour
            features.extend([
                np.sin(2 * np.pi * hour / 24),  # Cyclical hour encoding
                np.cos(2 * np.pi * hour / 24)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Behavior type features (one-hot encoding)
        behavior_types = ['crouching', 'loitering', 'erratic_movement', 
                         'suspicious_hand_movement', 'item_proximity']
        detected_behaviors = [b['type'] for b in behavior_data.get('behaviors', [])]
        
        for behavior_type in behavior_types:
            features.append(1 if behavior_type in detected_behaviors else 0)
        
        return np.array(features)
    
    def update_model(self, behavior_data: Dict, track_data: Dict):
        """
        Update the anomaly detection model with new data
        """
        features = self.extract_features(behavior_data, track_data)
        self.feature_history.append(features)
        
        # Keep only recent history (last 1000 samples)
        if len(self.feature_history) > 1000:
            self.feature_history = self.feature_history[-1000:]
        
        # Retrain model if we have enough samples
        if len(self.feature_history) >= 50:
            self._train_model()
    
    def detect_anomaly(self, behavior_data: Dict, track_data: Dict) -> Dict:
        """
        Detect if the current behavior is anomalous
        """
        features = self.extract_features(behavior_data, track_data)
        
        if not self.is_trained:
            # If model not trained, use rule-based approach
            return {
                'is_anomaly': behavior_data.get('suspicious_score', 0) > 0.7,
                'anomaly_score': behavior_data.get('suspicious_score', 0),
                'method': 'rule_based'
            }
        
        # Normalize features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict anomaly
        prediction = self.model.predict(features_scaled)[0]
        anomaly_score = self.model.decision_function(features_scaled)[0]
        
        # Convert to probability-like score (0-1 range)
        anomaly_score_normalized = max(0, min(1, (0.5 - anomaly_score) * 2))
        
        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': anomaly_score_normalized,
            'method': 'isolation_forest'
        }
    
    def _train_model(self):
        """
        Train the anomaly detection model
        """
        if len(self.feature_history) < 10:
            return
        
        try:
            # Convert to numpy array
            X = np.array(self.feature_history)
            
            # Handle any NaN or infinite values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Fit scaler and transform data
            X_scaled = self.scaler.fit_transform(X)
            
            # Train isolation forest
            self.model.fit(X_scaled)
            self.is_trained = True
            
            # Save model
            self._save_model()
            
        except Exception as e:
            print(f"Error training anomaly detection model: {e}")
    
    def _save_model(self):
        """
        Save the trained model and scaler
        """
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
        except Exception as e:
            print(f"Error saving anomaly detection model: {e}")
    
    def _load_model(self):
        """
        Load existing model and scaler if available
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.is_trained = True
                print("Loaded existing anomaly detection model")
                
        except Exception as e:
            print(f"Error loading anomaly detection model: {e}")
            self.is_trained = False
    
    def get_model_stats(self) -> Dict:
        """
        Get statistics about the anomaly detection model
        """
        return {
            'is_trained': self.is_trained,
            'training_samples': len(self.feature_history),
            'model_type': 'IsolationForest'
        }
