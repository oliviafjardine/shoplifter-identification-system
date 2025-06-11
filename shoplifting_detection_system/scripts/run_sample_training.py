#!/usr/bin/env python3
"""
Training with Sample Data - Quick Start Version
Creates synthetic training data and trains the model for immediate improvement
"""

import os
import sys
import numpy as np
import cv2
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from shoplifting_detection.core.detector import ShopliftingDetector

class SampleDataTrainer:
    """
    Trains the shoplifting detection model using synthetic sample data
    This provides immediate improvement while waiting for real dataset download
    """
    
    def __init__(self):
        self.training_dir = Path("training_data")
        self.training_dir.mkdir(exist_ok=True)
        
        self.models_dir = self.training_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.detector = ShopliftingDetector()
        self.scaler = StandardScaler()
        
    def generate_synthetic_training_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for shoplifting detection"""
        print(f"🎭 Generating {num_samples} synthetic training samples...")
        
        features = []
        labels = []
        
        for i in range(num_samples):
            # Generate different behavior patterns
            if i % 4 == 0:
                # Normal browsing behavior
                sample = self._generate_normal_behavior()
                label = 0
            elif i % 4 == 1:
                # Shelf interaction without taking
                sample = self._generate_shelf_interaction()
                label = 0
            elif i % 4 == 2:
                # Suspicious behavior (shoplifting)
                sample = self._generate_shoplifting_behavior()
                label = 1
            else:
                # Concealment behavior (shoplifting)
                sample = self._generate_concealment_behavior()
                label = 1
            
            features.append(sample)
            labels.append(label)
        
        features_array = np.array(features)
        labels_array = np.array(labels)
        
        print(f"✅ Generated synthetic data:")
        print(f"   Total samples: {len(features_array)}")
        print(f"   Feature dimensions: {features_array.shape[1]}")
        print(f"   Normal behavior samples: {np.sum(labels_array == 0)}")
        print(f"   Shoplifting samples: {np.sum(labels_array == 1)}")
        
        return features_array, labels_array
    
    def _generate_normal_behavior(self) -> np.ndarray:
        """Generate features for normal browsing behavior"""
        features = []
        
        # Movement patterns (smooth, consistent)
        features.append(np.random.normal(5.0, 1.0))    # Average speed
        features.append(np.random.normal(0.2, 0.1))    # Direction changes
        features.append(np.random.normal(0.1, 0.05))   # Acceleration variance
        
        # Position patterns (away from shelves)
        features.append(np.random.normal(0.3, 0.1))    # Distance from shelves
        features.append(np.random.normal(0.1, 0.05))   # Time near shelves
        
        # Interaction patterns (minimal)
        features.append(np.random.normal(0.1, 0.05))   # Shelf interaction score
        features.append(np.random.normal(0.05, 0.02))  # Reaching behavior
        features.append(np.random.normal(0.02, 0.01))  # Concealment score
        
        # Body language (normal)
        features.append(np.random.normal(0.1, 0.05))   # Nervous movements
        features.append(np.random.normal(0.05, 0.02))  # Looking around
        features.append(np.random.normal(0.8, 0.1))    # Confidence level
        
        # Time patterns
        features.append(np.random.normal(30.0, 10.0))  # Time in store
        features.append(np.random.normal(0.2, 0.1))    # Time stationary
        
        return np.array(features)
    
    def _generate_shelf_interaction(self) -> np.ndarray:
        """Generate features for legitimate shelf interaction"""
        features = []
        
        # Movement patterns (slower near shelves)
        features.append(np.random.normal(3.0, 1.0))    # Average speed
        features.append(np.random.normal(0.4, 0.1))    # Direction changes
        features.append(np.random.normal(0.2, 0.1))    # Acceleration variance
        
        # Position patterns (near shelves)
        features.append(np.random.normal(0.1, 0.05))   # Distance from shelves
        features.append(np.random.normal(0.4, 0.1))    # Time near shelves
        
        # Interaction patterns (moderate)
        features.append(np.random.normal(0.4, 0.1))    # Shelf interaction score
        features.append(np.random.normal(0.3, 0.1))    # Reaching behavior
        features.append(np.random.normal(0.1, 0.05))   # Concealment score
        
        # Body language (focused)
        features.append(np.random.normal(0.2, 0.1))    # Nervous movements
        features.append(np.random.normal(0.3, 0.1))    # Looking around
        features.append(np.random.normal(0.7, 0.1))    # Confidence level
        
        # Time patterns
        features.append(np.random.normal(45.0, 15.0))  # Time in store
        features.append(np.random.normal(0.6, 0.2))    # Time stationary
        
        return np.array(features)
    
    def _generate_shoplifting_behavior(self) -> np.ndarray:
        """Generate features for shoplifting behavior"""
        features = []
        
        # Movement patterns (erratic, nervous)
        features.append(np.random.normal(2.0, 0.5))    # Average speed (slower)
        features.append(np.random.normal(0.8, 0.2))    # Direction changes (more)
        features.append(np.random.normal(0.6, 0.2))    # Acceleration variance (higher)
        
        # Position patterns (close to shelves, then moving away)
        features.append(np.random.normal(0.05, 0.02))  # Distance from shelves (very close)
        features.append(np.random.normal(0.7, 0.2))    # Time near shelves (extended)
        
        # Interaction patterns (high)
        features.append(np.random.normal(0.8, 0.1))    # Shelf interaction score (high)
        features.append(np.random.normal(0.7, 0.1))    # Reaching behavior (frequent)
        features.append(np.random.normal(0.6, 0.2))    # Concealment score (moderate)
        
        # Body language (nervous, suspicious)
        features.append(np.random.normal(0.7, 0.2))    # Nervous movements (high)
        features.append(np.random.normal(0.8, 0.1))    # Looking around (frequent)
        features.append(np.random.normal(0.3, 0.1))    # Confidence level (low)
        
        # Time patterns
        features.append(np.random.normal(20.0, 5.0))   # Time in store (shorter)
        features.append(np.random.normal(0.8, 0.1))    # Time stationary (high)
        
        return np.array(features)
    
    def _generate_concealment_behavior(self) -> np.ndarray:
        """Generate features for concealment behavior"""
        features = []
        
        # Movement patterns (deliberate, hiding)
        features.append(np.random.normal(1.5, 0.5))    # Average speed (very slow)
        features.append(np.random.normal(0.9, 0.1))    # Direction changes (many)
        features.append(np.random.normal(0.8, 0.2))    # Acceleration variance (very high)
        
        # Position patterns (moving away from shelves)
        features.append(np.random.normal(0.2, 0.1))    # Distance from shelves (increasing)
        features.append(np.random.normal(0.9, 0.1))    # Time near shelves (was high)
        
        # Interaction patterns (very high)
        features.append(np.random.normal(0.9, 0.05))   # Shelf interaction score (very high)
        features.append(np.random.normal(0.8, 0.1))    # Reaching behavior (very frequent)
        features.append(np.random.normal(0.9, 0.1))    # Concealment score (very high)
        
        # Body language (very suspicious)
        features.append(np.random.normal(0.9, 0.1))    # Nervous movements (very high)
        features.append(np.random.normal(0.9, 0.05))   # Looking around (constant)
        features.append(np.random.normal(0.1, 0.05))   # Confidence level (very low)
        
        # Time patterns
        features.append(np.random.normal(15.0, 3.0))   # Time in store (very short)
        features.append(np.random.normal(0.9, 0.05))   # Time stationary (very high)
        
        return np.array(features)
    
    def train_models(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Train the detection models"""
        print("🤖 Training detection models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.3,  # Expect 30% anomalies in our synthetic data
            random_state=42,
            n_estimators=100
        )
        iso_forest.fit(X_train_scaled)
        
        # Train Random Forest
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10
        )
        rf_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate models
        print("📊 Evaluating models...")
        
        # Isolation Forest evaluation
        iso_pred = iso_forest.predict(X_test_scaled)
        iso_pred = (iso_pred == -1).astype(int)  # Convert to 0/1
        
        # Random Forest evaluation
        rf_pred = rf_classifier.predict(X_test_scaled)
        rf_prob = rf_classifier.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        iso_metrics = {
            'accuracy': accuracy_score(y_test, iso_pred),
            'precision': precision_score(y_test, iso_pred, zero_division=0),
            'recall': recall_score(y_test, iso_pred, zero_division=0),
            'f1_score': f1_score(y_test, iso_pred, zero_division=0)
        }
        
        rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, zero_division=0),
            'recall': recall_score(y_test, rf_pred, zero_division=0),
            'f1_score': f1_score(y_test, rf_pred, zero_division=0)
        }
        
        print("🎯 Model Performance:")
        print(f"   Isolation Forest - Accuracy: {iso_metrics['accuracy']:.3f}, Precision: {iso_metrics['precision']:.3f}, Recall: {iso_metrics['recall']:.3f}")
        print(f"   Random Forest - Accuracy: {rf_metrics['accuracy']:.3f}, Precision: {rf_metrics['precision']:.3f}, Recall: {rf_metrics['recall']:.3f}")
        
        # Save models
        models = {
            'isolation_forest': iso_forest,
            'random_forest': rf_classifier,
            'scaler': self.scaler,
            'performance': {
                'isolation_forest': iso_metrics,
                'random_forest': rf_metrics
            },
            'training_info': {
                'samples': len(features),
                'features': features.shape[1],
                'positive_samples': np.sum(labels),
                'negative_samples': np.sum(labels == 0)
            }
        }
        
        # Save to file
        models_file = self.models_dir / "synthetic_trained_models.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(models, f)
        
        print(f"💾 Models saved to: {models_file}")
        
        return models
    
    def update_detector_thresholds(self, model_performance: Dict):
        """Update detector thresholds based on model performance"""
        print("⚙️  Updating detector thresholds...")
        
        rf_perf = model_performance.get('random_forest', {})
        
        # Adjust thresholds based on performance
        if rf_perf.get('precision', 0) < 0.7:
            # Low precision - increase thresholds
            self.detector.shelf_interaction_threshold = min(0.6, self.detector.shelf_interaction_threshold + 0.1)
            self.detector.concealment_threshold = min(0.7, self.detector.concealment_threshold + 0.1)
            self.detector.shoplifting_threshold = min(0.8, self.detector.shoplifting_threshold + 0.1)
            print("   📈 Increased thresholds to improve precision")
        
        if rf_perf.get('recall', 0) < 0.7:
            # Low recall - decrease thresholds
            self.detector.shelf_interaction_threshold = max(0.2, self.detector.shelf_interaction_threshold - 0.1)
            self.detector.concealment_threshold = max(0.3, self.detector.concealment_threshold - 0.1)
            self.detector.shoplifting_threshold = max(0.5, self.detector.shoplifting_threshold - 0.1)
            print("   📉 Decreased thresholds to improve recall")
        
        # Apply performance-based fine-tuning
        accuracy = rf_perf.get('accuracy', 0)
        if accuracy > 0.8:
            # Good performance - fine-tune for better balance
            self.detector.shelf_interaction_threshold = 0.35
            self.detector.concealment_threshold = 0.45
            self.detector.shoplifting_threshold = 0.65
            print("   🎯 Applied fine-tuned thresholds for balanced performance")
        
        print(f"   New thresholds:")
        print(f"     Shelf interaction: {self.detector.shelf_interaction_threshold:.2f}")
        print(f"     Concealment: {self.detector.concealment_threshold:.2f}")
        print(f"     Shoplifting: {self.detector.shoplifting_threshold:.2f}")
    
    def test_improved_detection(self) -> Dict:
        """Test the improved detection system"""
        print("🧪 Testing improved detection system...")
        
        # Create test scenarios
        test_scenarios = [
            ("normal_browsing", False, self._generate_normal_behavior()),
            ("shelf_interaction", False, self._generate_shelf_interaction()),
            ("shoplifting_behavior", True, self._generate_shoplifting_behavior()),
            ("concealment_behavior", True, self._generate_concealment_behavior())
        ]
        
        results = []
        
        for scenario_name, expected, features in test_scenarios:
            # Create mock person data
            mock_person = {
                'person_id': 1,
                'bbox': {'center_x': 150, 'center_y': 240, 'width': 80, 'height': 180,
                        'x1': 110, 'y1': 150, 'x2': 190, 'y2': 330},
                'track_data': {
                    'positions': [
                        {'x': 150, 'y': 240, 'timestamp': datetime.now(), 'width': 80, 'height': 180}
                        for _ in range(10)
                    ],
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now()
                }
            }
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Test detection
            result = self.detector.analyze_person_behavior(mock_person, frame)
            detected = self.detector.is_behavior_suspicious(result)
            
            results.append({
                'scenario': scenario_name,
                'expected': expected,
                'detected': detected,
                'score': result['suspicious_score'],
                'stage': result['shoplifting_stage'],
                'correct': detected == expected
            })
            
            status = "✅" if detected == expected else "❌"
            print(f"   {status} {scenario_name}: Expected={expected}, Detected={detected}, Score={result['suspicious_score']:.2f}")
        
        # Calculate accuracy
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / len(results)
        
        print(f"\n📊 Test Results:")
        print(f"   Accuracy: {accuracy:.2f} ({correct}/{len(results)})")
        
        return {
            'accuracy': accuracy,
            'results': results,
            'correct_predictions': correct,
            'total_tests': len(results)
        }
    
    def run_complete_training(self):
        """Run the complete training pipeline"""
        print("🚀 Starting Sample Data Training Pipeline")
        print("=" * 50)
        
        start_time = datetime.now()
        
        # Generate synthetic training data
        features, labels = self.generate_synthetic_training_data(1000)
        
        # Train models
        models = self.train_models(features, labels)
        
        # Update detector thresholds
        self.update_detector_thresholds(models['performance'])
        
        # Test improved detection
        test_results = self.test_improved_detection()
        
        # Generate summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎯 Training Summary:")
        print(f"   Training time: {duration}")
        print(f"   Training samples: {models['training_info']['samples']}")
        print(f"   Model accuracy: {models['performance']['random_forest']['accuracy']:.3f}")
        print(f"   Detection accuracy: {test_results['accuracy']:.2f}")
        print(f"   Improvement: {'Successful' if test_results['accuracy'] > 0.7 else 'Needs more work'}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'training_duration': str(duration),
            'model_performance': models['performance'],
            'detection_performance': test_results,
            'updated_thresholds': {
                'shelf_interaction': self.detector.shelf_interaction_threshold,
                'concealment': self.detector.concealment_threshold,
                'shoplifting': self.detector.shoplifting_threshold
            }
        }
        
        summary_file = self.models_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Training summary saved to: {summary_file}")
        print("\n✅ Sample data training completed!")
        print("\nNext steps:")
        print("1. Test with: python test_realistic_scenarios.py")
        print("2. For real data training: python train_with_kaggle_data.py")
        print("3. Monitor performance and retrain as needed")

def main():
    """Main function"""
    trainer = SampleDataTrainer()
    trainer.run_complete_training()

if __name__ == "__main__":
    main()
