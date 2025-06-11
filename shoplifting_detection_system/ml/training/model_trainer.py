#!/usr/bin/env python3
"""
Model Trainer for Shoplifting Detection
Uses real shoplifting video data to train and improve the detection model
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pickle
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from detection.shoplifting_detector import ShopliftingDetector
from detection.anomaly_detector import AnomalyDetector
from training.dataset_manager import ShopliftingDatasetManager

class ShopliftingModelTrainer:
    """
    Trains shoplifting detection models using real video data
    """
    
    def __init__(self, training_data_dir: str = "training_data"):
        self.training_data_dir = Path(training_data_dir)
        self.models_dir = self.training_data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.detector = ShopliftingDetector()
        self.anomaly_detector = AnomalyDetector()
        
        self.training_features = []
        self.training_labels = []
        self.feature_scaler = StandardScaler()
        
    def extract_features_from_frame(self, frame: np.ndarray, person_bbox: Optional[Dict] = None) -> np.ndarray:
        """
        Extract features from a video frame for training
        """
        features = []
        
        # Basic frame features
        height, width = frame.shape[:2]
        features.extend([height, width])
        
        # Color features
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features.append(np.mean(gray))  # Average brightness
        features.append(np.std(gray))   # Brightness variation
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / (height * width))  # Edge density
        
        # Motion features (simplified - would need optical flow for real motion)
        # For now, use texture features as proxy
        features.append(cv2.Laplacian(gray, cv2.CV_64F).var())  # Texture variance
        
        # If person bbox is provided, extract person-specific features
        if person_bbox:
            x1, y1, x2, y2 = person_bbox['x1'], person_bbox['y1'], person_bbox['x2'], person_bbox['y2']
            
            # Ensure bbox is within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            if x2 > x1 and y2 > y1:
                person_region = frame[y1:y2, x1:x2]
                person_gray = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
                
                # Person-specific features
                features.append(person_region.shape[0])  # Height
                features.append(person_region.shape[1])  # Width
                features.append(np.mean(person_gray))    # Person brightness
                features.append(np.std(person_gray))     # Person brightness variation
                
                # Person position relative to frame
                features.append(x1 / width)   # Relative x position
                features.append(y1 / height)  # Relative y position
            else:
                # Add zeros if bbox is invalid
                features.extend([0, 0, 0, 0, 0, 0])
        else:
            # Add zeros if no person bbox
            features.extend([0, 0, 0, 0, 0, 0])
        
        return np.array(features)
    
    def process_training_videos(self, max_frames_per_video: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process training videos to extract features and labels
        """
        print("🎬 Processing training videos...")
        
        frames_dir = self.training_data_dir / "processed" / "frames"
        annotations_file = self.training_data_dir / "processed" / "training_annotations.json"
        
        if not frames_dir.exists() or not annotations_file.exists():
            print("❌ Training data not found. Run dataset_manager.py first.")
            return np.array([]), np.array([])
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        features_list = []
        labels_list = []
        
        for video_name, video_data in annotations['annotations'].items():
            print(f"  📹 Processing {video_name}...")
            
            video_frames_dir = frames_dir / video_name
            if not video_frames_dir.exists():
                continue
            
            frame_files = sorted(list(video_frames_dir.glob("*.jpg")))
            
            # Limit frames per video to avoid memory issues
            if len(frame_files) > max_frames_per_video:
                step = len(frame_files) // max_frames_per_video
                frame_files = frame_files[::step][:max_frames_per_video]
            
            for frame_file in frame_files:
                try:
                    # Load frame
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        continue
                    
                    # Extract features
                    features = self.extract_features_from_frame(frame)
                    features_list.append(features)
                    
                    # Label (1 for shoplifting, 0 for normal)
                    label = 1 if video_data['is_anomaly'] else 0
                    labels_list.append(label)
                    
                except Exception as e:
                    print(f"    ❌ Error processing {frame_file}: {e}")
                    continue
            
            print(f"    ✅ Processed {len(frame_files)} frames")
        
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        print(f"✅ Extracted features from {len(features_array)} frames")
        print(f"   Feature dimensions: {features_array.shape}")
        print(f"   Positive samples (shoplifting): {np.sum(labels_array)}")
        print(f"   Negative samples (normal): {len(labels_array) - np.sum(labels_array)}")
        
        return features_array, labels_array
    
    def train_anomaly_detector(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Train the anomaly detection model
        """
        print("🤖 Training anomaly detection model...")
        
        if len(features) == 0:
            print("❌ No training data available")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Train Isolation Forest for anomaly detection
        isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        isolation_forest.fit(X_train_scaled)
        
        # Train Random Forest for classification
        random_forest = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        random_forest.fit(X_train_scaled, y_train)
        
        # Evaluate models
        print("📊 Evaluating models...")
        
        # Isolation Forest evaluation
        iso_predictions = isolation_forest.predict(X_test_scaled)
        iso_predictions = (iso_predictions == -1).astype(int)  # Convert to 0/1
        
        # Random Forest evaluation
        rf_predictions = random_forest.predict(X_test_scaled)
        rf_probabilities = random_forest.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        iso_metrics = {
            'accuracy': accuracy_score(y_test, iso_predictions),
            'precision': precision_score(y_test, iso_predictions, zero_division=0),
            'recall': recall_score(y_test, iso_predictions, zero_division=0),
            'f1_score': f1_score(y_test, iso_predictions, zero_division=0)
        }
        
        rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_predictions),
            'precision': precision_score(y_test, rf_predictions, zero_division=0),
            'recall': recall_score(y_test, rf_predictions, zero_division=0),
            'f1_score': f1_score(y_test, rf_predictions, zero_division=0)
        }
        
        print("🎯 Isolation Forest Results:")
        for metric, value in iso_metrics.items():
            print(f"   {metric}: {value:.3f}")
        
        print("🎯 Random Forest Results:")
        for metric, value in rf_metrics.items():
            print(f"   {metric}: {value:.3f}")
        
        # Save models
        models = {
            'isolation_forest': isolation_forest,
            'random_forest': random_forest,
            'feature_scaler': self.feature_scaler,
            'training_info': {
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_dimensions': features.shape[1],
                'positive_samples': np.sum(labels),
                'negative_samples': len(labels) - np.sum(labels)
            },
            'performance': {
                'isolation_forest': iso_metrics,
                'random_forest': rf_metrics
            }
        }
        
        # Save to file
        models_file = self.models_dir / "trained_models.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(models, f)
        
        print(f"💾 Models saved to: {models_file}")
        
        return models
    
    def update_detection_system(self, trained_models: Dict) -> bool:
        """
        Update the detection system with trained models
        """
        print("🔄 Updating detection system with trained models...")
        
        try:
            # Update anomaly detector
            if 'isolation_forest' in trained_models:
                self.anomaly_detector.model = trained_models['isolation_forest']
                self.anomaly_detector.scaler = trained_models['feature_scaler']
                self.anomaly_detector.is_trained = True
                
                # Save updated anomaly detector
                anomaly_model_file = self.models_dir / "anomaly_model.pkl"
                with open(anomaly_model_file, 'wb') as f:
                    pickle.dump(self.anomaly_detector.model, f)
                
                anomaly_scaler_file = self.models_dir / "anomaly_scaler.pkl"
                with open(anomaly_scaler_file, 'wb') as f:
                    pickle.dump(self.anomaly_detector.scaler, f)
                
                print("✅ Anomaly detector updated")
            
            # Update shoplifting detector with improved thresholds based on training results
            rf_performance = trained_models.get('performance', {}).get('random_forest', {})
            if rf_performance:
                # Adjust thresholds based on performance
                if rf_performance.get('precision', 0) < 0.7:
                    # Low precision - increase thresholds to reduce false positives
                    self.detector.shelf_interaction_threshold = min(0.6, self.detector.shelf_interaction_threshold + 0.1)
                    self.detector.concealment_threshold = min(0.7, self.detector.concealment_threshold + 0.1)
                    print("📈 Increased thresholds to improve precision")
                
                if rf_performance.get('recall', 0) < 0.7:
                    # Low recall - decrease thresholds to catch more incidents
                    self.detector.shelf_interaction_threshold = max(0.2, self.detector.shelf_interaction_threshold - 0.05)
                    self.detector.concealment_threshold = max(0.3, self.detector.concealment_threshold - 0.05)
                    print("📉 Decreased thresholds to improve recall")
            
            print("✅ Detection system updated successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error updating detection system: {e}")
            return False
    
    def train_complete_system(self) -> Dict:
        """
        Complete training pipeline
        """
        print("🚀 Starting complete training pipeline...")
        
        # Process training videos
        features, labels = self.process_training_videos()
        
        if len(features) == 0:
            print("❌ No training data available")
            return {}
        
        # Train models
        trained_models = self.train_anomaly_detector(features, labels)
        
        # Update detection system
        self.update_detection_system(trained_models)
        
        # Save training summary
        training_summary = {
            'timestamp': datetime.now().isoformat(),
            'training_data': {
                'total_samples': len(features),
                'positive_samples': int(np.sum(labels)),
                'negative_samples': int(len(labels) - np.sum(labels)),
                'feature_dimensions': features.shape[1]
            },
            'model_performance': trained_models.get('performance', {}),
            'updated_thresholds': {
                'shelf_interaction_threshold': self.detector.shelf_interaction_threshold,
                'concealment_threshold': self.detector.concealment_threshold,
                'shoplifting_threshold': self.detector.shoplifting_threshold
            }
        }
        
        summary_file = self.models_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"📋 Training summary saved to: {summary_file}")
        print("✅ Training pipeline completed!")
        
        return training_summary

def main():
    """Main training function"""
    print("🎯 Shoplifting Detection Model Trainer")
    print("=" * 50)
    
    # Check if training data exists
    training_data_dir = Path("training_data")
    if not training_data_dir.exists():
        print("❌ Training data directory not found.")
        print("Please run 'python training/dataset_manager.py' first to download and prepare the dataset.")
        return
    
    # Initialize trainer
    trainer = ShopliftingModelTrainer()
    
    # Run complete training pipeline
    summary = trainer.train_complete_system()
    
    if summary:
        print("\n📊 Training Summary:")
        print(f"   Total samples: {summary['training_data']['total_samples']}")
        print(f"   Positive samples: {summary['training_data']['positive_samples']}")
        print(f"   Feature dimensions: {summary['training_data']['feature_dimensions']}")
        
        if 'random_forest' in summary.get('model_performance', {}):
            rf_perf = summary['model_performance']['random_forest']
            print(f"   Model accuracy: {rf_perf.get('accuracy', 0):.3f}")
            print(f"   Model precision: {rf_perf.get('precision', 0):.3f}")
            print(f"   Model recall: {rf_perf.get('recall', 0):.3f}")
        
        print("\n🎯 Next steps:")
        print("1. Run 'python test_realistic_scenarios.py' to test improved detection")
        print("2. Use 'python training/evaluate_model.py' for detailed evaluation")
        print("3. Monitor real-world performance and retrain as needed")
    else:
        print("❌ Training failed. Please check the dataset and try again.")

if __name__ == "__main__":
    main()
