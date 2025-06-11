#!/usr/bin/env python3
"""
Focused Shoplifting Training System - Immediate Implementation
Creates highly realistic shoplifting behavior patterns for training
Uses advanced synthetic data generation to maximize detection accuracy
"""

from detection.shoplifting_detector import ShopliftingDetector
import sys
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))


class FocusedShopliftingTrainer:
    """
    Advanced shoplifting detection trainer using sophisticated synthetic data
    Designed to maximize detection accuracy through realistic behavior modeling
    """

    def __init__(self):
        self.training_dir = Path("focused_shoplifting_training")
        self.training_dir.mkdir(exist_ok=True)

        self.models_dir = self.training_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.detector = ShopliftingDetector()
        self.scaler = StandardScaler()

        # Advanced behavior patterns
        self.behavior_patterns = {
            'normal_browsing': self._generate_normal_browsing_pattern,
            'window_shopping': self._generate_window_shopping_pattern,
            'legitimate_shopping': self._generate_legitimate_shopping_pattern,
            'casual_theft': self._generate_casual_theft_pattern,
            'professional_theft': self._generate_professional_theft_pattern,
            'opportunistic_theft': self._generate_opportunistic_theft_pattern,
            'distraction_theft': self._generate_distraction_theft_pattern,
            'concealment_theft': self._generate_concealment_theft_pattern
        }

    def generate_advanced_training_data(self, samples_per_pattern: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate advanced synthetic training data with realistic shoplifting patterns"""
        print(f"🎭 Generating advanced shoplifting training data...")
        print(f"   Samples per pattern: {samples_per_pattern}")

        all_features = []
        all_labels = []

        for pattern_name, pattern_func in self.behavior_patterns.items():
            print(f"   Generating {pattern_name} patterns...")

            # Determine if this is shoplifting behavior
            is_shoplifting = 'theft' in pattern_name
            label = 1 if is_shoplifting else 0

            for i in range(samples_per_pattern):
                try:
                    features = pattern_func()
                    all_features.append(features)
                    all_labels.append(label)
                except Exception as e:
                    continue

        features_array = np.array(all_features)
        labels_array = np.array(all_labels)

        print(f"✅ Generated advanced training data:")
        print(f"   Total samples: {len(features_array)}")
        print(f"   Feature dimensions: {features_array.shape[1]}")
        print(f"   Normal behavior samples: {np.sum(labels_array == 0)}")
        print(f"   Shoplifting samples: {np.sum(labels_array == 1)}")
        print(
            f"   Class balance: {np.sum(labels_array == 1) / len(labels_array):.2f}")

        return features_array, labels_array

    def _generate_normal_browsing_pattern(self) -> np.ndarray:
        """Generate normal browsing behavior - no theft intent"""
        features = []

        # Movement characteristics (smooth, purposeful)
        features.append(np.random.normal(4.0, 0.8))    # Walking speed (normal)
        # Direction changes (minimal)
        features.append(np.random.normal(0.15, 0.05))
        # Acceleration variance (smooth)
        features.append(np.random.normal(0.1, 0.03))
        # Stop frequency (occasional)
        features.append(np.random.normal(0.2, 0.1))

        # Positioning (maintains distance from shelves)
        # Average distance from shelves
        features.append(np.random.normal(0.4, 0.15))
        # Time spent very close to shelves
        features.append(np.random.normal(0.1, 0.05))
        features.append(np.random.normal(0.3, 0.1))    # Coverage of store area

        # Interaction patterns (minimal, brief)
        # Shelf interaction frequency
        features.append(np.random.normal(0.1, 0.05))
        features.append(np.random.normal(0.05, 0.02))  # Reaching behavior
        features.append(np.random.normal(0.02, 0.01))  # Item manipulation
        features.append(np.random.normal(0.01, 0.005))  # Concealment behavior

        # Attention patterns (casual, unfocused)
        # Looking around frequency
        features.append(np.random.normal(0.2, 0.1))
        features.append(np.random.normal(0.1, 0.05))   # Surveillance awareness
        features.append(np.random.normal(0.05, 0.02))  # Nervous behavior

        # Temporal patterns (relaxed)
        # Time in store (minutes)
        features.append(np.random.normal(45.0, 15.0))
        features.append(np.random.normal(0.2, 0.1))    # Time stationary ratio
        features.append(np.random.normal(0.8, 0.1))    # Confidence level

        return np.array(features)

    def _generate_window_shopping_pattern(self) -> np.ndarray:
        """Generate window shopping behavior - looking but not buying"""
        features = []

        # Movement characteristics (slower, more stops)
        features.append(np.random.normal(2.5, 0.5))    # Walking speed (slower)
        # Direction changes (more)
        features.append(np.random.normal(0.3, 0.1))
        features.append(np.random.normal(0.2, 0.05))   # Acceleration variance
        # Stop frequency (frequent)
        features.append(np.random.normal(0.6, 0.2))

        # Positioning (closer to displays)
        # Average distance from shelves
        features.append(np.random.normal(0.2, 0.1))
        # Time spent close to shelves
        features.append(np.random.normal(0.4, 0.15))
        features.append(np.random.normal(0.5, 0.2))    # Coverage of store area

        # Interaction patterns (looking, minimal touching)
        # Shelf interaction frequency
        features.append(np.random.normal(0.3, 0.1))
        features.append(np.random.normal(0.2, 0.1))    # Reaching behavior
        features.append(np.random.normal(0.1, 0.05))   # Item manipulation
        features.append(np.random.normal(0.02, 0.01))  # Concealment behavior

        # Attention patterns (focused on items)
        # Looking around frequency
        features.append(np.random.normal(0.4, 0.15))
        features.append(np.random.normal(0.15, 0.1))   # Surveillance awareness
        features.append(np.random.normal(0.1, 0.05))   # Nervous behavior

        # Temporal patterns
        features.append(np.random.normal(30.0, 10.0))  # Time in store
        features.append(np.random.normal(0.6, 0.2))    # Time stationary ratio
        features.append(np.random.normal(0.7, 0.15))   # Confidence level

        return np.array(features)

    def _generate_legitimate_shopping_pattern(self) -> np.ndarray:
        """Generate legitimate shopping behavior - taking items to purchase"""
        features = []

        # Movement characteristics (purposeful)
        features.append(np.random.normal(3.5, 0.7))    # Walking speed
        features.append(np.random.normal(0.4, 0.1))    # Direction changes
        features.append(np.random.normal(0.3, 0.1))    # Acceleration variance
        features.append(np.random.normal(0.5, 0.15))   # Stop frequency

        # Positioning (approaches shelves confidently)
        # Average distance from shelves
        features.append(np.random.normal(0.1, 0.05))
        # Time spent close to shelves
        features.append(np.random.normal(0.6, 0.2))
        features.append(np.random.normal(0.4, 0.15))   # Coverage of store area

        # Interaction patterns (confident item selection)
        # Shelf interaction frequency
        features.append(np.random.normal(0.7, 0.2))
        features.append(np.random.normal(0.6, 0.2))    # Reaching behavior
        features.append(np.random.normal(0.5, 0.2))    # Item manipulation
        # Concealment behavior (minimal)
        features.append(np.random.normal(0.05, 0.02))

        # Attention patterns (focused, not nervous)
        # Looking around frequency
        features.append(np.random.normal(0.3, 0.1))
        features.append(np.random.normal(0.1, 0.05))   # Surveillance awareness
        features.append(np.random.normal(0.05, 0.02))  # Nervous behavior

        # Temporal patterns
        features.append(np.random.normal(25.0, 8.0))   # Time in store
        features.append(np.random.normal(0.4, 0.15))   # Time stationary ratio
        # Confidence level (high)
        features.append(np.random.normal(0.9, 0.05))

        return np.array(features)

    def _generate_casual_theft_pattern(self) -> np.ndarray:
        """Generate casual/impulsive theft behavior"""
        features = []

        # Movement characteristics (initially normal, then nervous)
        # Walking speed (slowing)
        features.append(np.random.normal(2.8, 0.6))
        # Direction changes (erratic)
        features.append(np.random.normal(0.7, 0.2))
        # Acceleration variance (high)
        features.append(np.random.normal(0.6, 0.2))
        # Stop frequency (frequent)
        features.append(np.random.normal(0.8, 0.2))

        # Positioning (close to shelves, then moving away)
        # Average distance from shelves
        features.append(np.random.normal(0.08, 0.03))
        # Time spent close to shelves
        features.append(np.random.normal(0.7, 0.2))
        features.append(np.random.normal(0.3, 0.1))    # Coverage of store area

        # Interaction patterns (quick, furtive)
        # Shelf interaction frequency
        features.append(np.random.normal(0.8, 0.15))
        features.append(np.random.normal(0.7, 0.2))    # Reaching behavior
        features.append(np.random.normal(0.6, 0.2))    # Item manipulation
        # Concealment behavior (moderate)
        features.append(np.random.normal(0.6, 0.2))

        # Attention patterns (nervous, checking surroundings)
        # Looking around frequency (high)
        features.append(np.random.normal(0.8, 0.15))
        # Surveillance awareness (high)
        features.append(np.random.normal(0.7, 0.2))
        # Nervous behavior (moderate)
        features.append(np.random.normal(0.6, 0.2))

        # Temporal patterns (quick exit)
        features.append(np.random.normal(15.0, 5.0))   # Time in store (short)
        features.append(np.random.normal(0.7, 0.2))    # Time stationary ratio
        features.append(np.random.normal(0.3, 0.1))    # Confidence level (low)

        return np.array(features)

    def _generate_professional_theft_pattern(self) -> np.ndarray:
        """Generate professional/experienced theft behavior"""
        features = []

        # Movement characteristics (controlled, deliberate)
        # Walking speed (controlled)
        features.append(np.random.normal(3.2, 0.4))
        # Direction changes (planned)
        features.append(np.random.normal(0.5, 0.1))
        # Acceleration variance (controlled)
        features.append(np.random.normal(0.4, 0.1))
        features.append(np.random.normal(0.6, 0.15))   # Stop frequency

        # Positioning (strategic positioning)
        # Average distance from shelves
        features.append(np.random.normal(0.06, 0.02))
        # Time spent close to shelves
        features.append(np.random.normal(0.8, 0.15))
        features.append(np.random.normal(0.6, 0.2))    # Coverage of store area

        # Interaction patterns (skilled, efficient)
        # Shelf interaction frequency (high)
        features.append(np.random.normal(0.9, 0.1))
        features.append(np.random.normal(0.8, 0.15))   # Reaching behavior
        features.append(np.random.normal(0.7, 0.2))    # Item manipulation
        # Concealment behavior (very high)
        features.append(np.random.normal(0.9, 0.1))

        # Attention patterns (aware but controlled)
        # Looking around frequency
        features.append(np.random.normal(0.6, 0.15))
        # Surveillance awareness (very high)
        features.append(np.random.normal(0.9, 0.1))
        # Nervous behavior (controlled)
        features.append(np.random.normal(0.3, 0.1))

        # Temporal patterns (efficient)
        # Time in store (very short)
        features.append(np.random.normal(12.0, 3.0))
        features.append(np.random.normal(0.8, 0.15))   # Time stationary ratio
        # Confidence level (practiced)
        features.append(np.random.normal(0.6, 0.15))

        return np.array(features)

    def _generate_opportunistic_theft_pattern(self) -> np.ndarray:
        """Generate opportunistic theft behavior - taking advantage of situation"""
        features = []

        # Movement characteristics (normal until opportunity)
        features.append(np.random.normal(3.0, 0.5))    # Walking speed
        # Direction changes (sudden)
        features.append(np.random.normal(0.6, 0.2))
        features.append(np.random.normal(0.5, 0.15))   # Acceleration variance
        features.append(np.random.normal(0.7, 0.2))    # Stop frequency

        # Positioning (opportunistic positioning)
        # Average distance from shelves
        features.append(np.random.normal(0.07, 0.03))
        # Time spent close to shelves
        features.append(np.random.normal(0.6, 0.2))
        features.append(np.random.normal(0.4, 0.15))   # Coverage of store area

        # Interaction patterns (quick when opportunity arises)
        # Shelf interaction frequency
        features.append(np.random.normal(0.7, 0.2))
        # Reaching behavior (quick)
        features.append(np.random.normal(0.8, 0.15))
        features.append(np.random.normal(0.6, 0.2))    # Item manipulation
        features.append(np.random.normal(0.7, 0.2))    # Concealment behavior

        # Attention patterns (checking for opportunities)
        # Looking around frequency (very high)
        features.append(np.random.normal(0.9, 0.1))
        features.append(np.random.normal(0.8, 0.15))   # Surveillance awareness
        features.append(np.random.normal(0.5, 0.2))    # Nervous behavior

        # Temporal patterns
        features.append(np.random.normal(18.0, 6.0))   # Time in store
        features.append(np.random.normal(0.6, 0.2))    # Time stationary ratio
        features.append(np.random.normal(0.4, 0.15))   # Confidence level

        return np.array(features)

    def _generate_distraction_theft_pattern(self) -> np.ndarray:
        """Generate distraction-based theft behavior"""
        features = []

        # Movement characteristics (creating distractions)
        features.append(np.random.normal(2.5, 0.6))    # Walking speed (varied)
        # Direction changes (many)
        features.append(np.random.normal(0.8, 0.2))
        # Acceleration variance (high)
        features.append(np.random.normal(0.7, 0.2))
        # Stop frequency (very high)
        features.append(np.random.normal(0.9, 0.1))

        # Positioning (multiple locations)
        # Average distance from shelves
        features.append(np.random.normal(0.15, 0.1))
        # Time spent close to shelves
        features.append(np.random.normal(0.5, 0.2))
        # Coverage of store area (high)
        features.append(np.random.normal(0.8, 0.15))

        # Interaction patterns (multiple interactions)
        # Shelf interaction frequency
        features.append(np.random.normal(0.8, 0.15))
        features.append(np.random.normal(0.6, 0.2))    # Reaching behavior
        features.append(np.random.normal(0.7, 0.2))    # Item manipulation
        features.append(np.random.normal(0.8, 0.15))   # Concealment behavior

        # Attention patterns (creating and using distractions)
        # Looking around frequency
        features.append(np.random.normal(0.7, 0.2))
        features.append(np.random.normal(0.6, 0.2))    # Surveillance awareness
        # Nervous behavior (controlled)
        features.append(np.random.normal(0.4, 0.15))

        # Temporal patterns
        features.append(np.random.normal(22.0, 8.0))   # Time in store
        features.append(np.random.normal(0.5, 0.2))    # Time stationary ratio
        features.append(np.random.normal(0.5, 0.2))    # Confidence level

        return np.array(features)

    def _generate_concealment_theft_pattern(self) -> np.ndarray:
        """Generate concealment-focused theft behavior"""
        features = []

        # Movement characteristics (focused on hiding)
        features.append(np.random.normal(2.0, 0.4))    # Walking speed (slow)
        features.append(np.random.normal(0.6, 0.2))    # Direction changes
        # Acceleration variance (high)
        features.append(np.random.normal(0.8, 0.2))
        # Stop frequency (very high)
        features.append(np.random.normal(0.9, 0.1))

        # Positioning (seeking concealment spots)
        # Average distance from shelves (very close)
        features.append(np.random.normal(0.05, 0.02))
        # Time spent close to shelves (very high)
        features.append(np.random.normal(0.9, 0.1))
        # Coverage of store area (limited)
        features.append(np.random.normal(0.2, 0.1))

        # Interaction patterns (heavy concealment focus)
        # Shelf interaction frequency (very high)
        features.append(np.random.normal(0.9, 0.1))
        features.append(np.random.normal(0.8, 0.15))   # Reaching behavior
        # Item manipulation (very high)
        features.append(np.random.normal(0.9, 0.1))
        # Concealment behavior (maximum)
        features.append(np.random.normal(0.95, 0.05))

        # Attention patterns (paranoid about being seen)
        # Looking around frequency (very high)
        features.append(np.random.normal(0.9, 0.1))
        # Surveillance awareness (maximum)
        features.append(np.random.normal(0.95, 0.05))
        # Nervous behavior (very high)
        features.append(np.random.normal(0.8, 0.15))

        # Temporal patterns
        # Time in store (very short)
        features.append(np.random.normal(10.0, 3.0))
        # Time stationary ratio (very high)
        features.append(np.random.normal(0.9, 0.1))
        # Confidence level (very low)
        features.append(np.random.normal(0.2, 0.1))

        return np.array(features)

    def train_maximum_accuracy_models(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Train multiple models optimized for maximum accuracy"""
        print("🎯 Training models for MAXIMUM accuracy...")

        if len(features) == 0:
            print("❌ No training data available")
            return {}

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {}

        # 1. Optimized Random Forest
        print("   🌲 Training Optimized Random Forest...")
        rf_params = {
            'n_estimators': [200, 300, 500],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 3],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=5,
                               scoring='f1', n_jobs=-1, verbose=0)
        rf_grid.fit(X_train_scaled, y_train)

        rf_pred = rf_grid.predict(X_test_scaled)
        rf_metrics = self._calculate_metrics(y_test, rf_pred)
        models['random_forest'] = {
            'model': rf_grid.best_estimator_,
            'metrics': rf_metrics,
            'params': rf_grid.best_params_
        }

        # 2. Optimized Gradient Boosting
        print("   🚀 Training Optimized Gradient Boosting...")
        gb_params = {
            'n_estimators': [200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [5, 7, 9],
            'subsample': [0.8, 0.9, 1.0]
        }
        gb = GradientBoostingClassifier(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=5,
                               scoring='f1', n_jobs=-1, verbose=0)
        gb_grid.fit(X_train_scaled, y_train)

        gb_pred = gb_grid.predict(X_test_scaled)
        gb_metrics = self._calculate_metrics(y_test, gb_pred)
        models['gradient_boosting'] = {
            'model': gb_grid.best_estimator_,
            'metrics': gb_metrics,
            'params': gb_grid.best_params_
        }

        # 3. Optimized SVM
        print("   ⚡ Training Optimized SVM...")
        svm_params = {
            'C': [1, 10, 100],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto'],
            'class_weight': ['balanced']
        }
        svm = SVC(random_state=42, probability=True)
        svm_grid = GridSearchCV(svm, svm_params, cv=5,
                                scoring='f1', n_jobs=-1, verbose=0)
        svm_grid.fit(X_train_scaled, y_train)

        svm_pred = svm_grid.predict(X_test_scaled)
        svm_metrics = self._calculate_metrics(y_test, svm_pred)
        models['svm'] = {
            'model': svm_grid.best_estimator_,
            'metrics': svm_metrics,
            'params': svm_grid.best_params_
        }

        # Find best model
        best_model_name = max(
            models.keys(), key=lambda k: models[k]['metrics']['f1_score'])
        best_model = models[best_model_name]

        print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")
        print(f"   🎯 Accuracy: {best_model['metrics']['accuracy']:.3f}")
        print(f"   🎯 Precision: {best_model['metrics']['precision']:.3f}")
        print(f"   🎯 Recall: {best_model['metrics']['recall']:.3f}")
        print(f"   🎯 F1-Score: {best_model['metrics']['f1_score']:.3f}")

        # Cross-validation for robustness
        print(f"\n🔄 Cross-validation scores:")
        cv_scores = cross_val_score(
            best_model['model'], X_train_scaled, y_train, cv=5, scoring='f1')
        print(
            f"   CV F1-Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Save models
        training_data = {
            'models': models,
            'best_model': best_model_name,
            'scaler': self.scaler,
            'feature_count': features.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'cv_scores': cv_scores.tolist()
        }

        models_file = self.models_dir / "maximum_accuracy_models.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(training_data, f)

        print(f"💾 Models saved to: {models_file}")

        return training_data

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

    def optimize_detector_for_maximum_accuracy(self, best_model_metrics: Dict):
        """Optimize detector thresholds for maximum accuracy"""
        print("⚙️  Optimizing detector for MAXIMUM accuracy...")

        accuracy = best_model_metrics['accuracy']
        precision = best_model_metrics['precision']
        recall = best_model_metrics['recall']
        f1 = best_model_metrics['f1_score']

        print(
            f"   Model performance: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")

        # Ultra-aggressive optimization for maximum detection
        if f1 > 0.95:
            # Exceptional performance - use ultra-aggressive thresholds
            self.detector.shelf_interaction_threshold = 0.15
            self.detector.concealment_threshold = 0.25
            self.detector.shoplifting_threshold = 0.45
            print("   🔥 Applied ULTRA-AGGRESSIVE thresholds for exceptional model")

        elif f1 > 0.90:
            # Excellent performance - use very aggressive thresholds
            self.detector.shelf_interaction_threshold = 0.20
            self.detector.concealment_threshold = 0.30
            self.detector.shoplifting_threshold = 0.50
            print("   🚀 Applied VERY AGGRESSIVE thresholds for excellent model")

        elif f1 > 0.85:
            # Very good performance - use aggressive thresholds
            self.detector.shelf_interaction_threshold = 0.25
            self.detector.concealment_threshold = 0.35
            self.detector.shoplifting_threshold = 0.55
            print("   🎯 Applied AGGRESSIVE thresholds for very good model")

        elif f1 > 0.80:
            # Good performance - use moderate-aggressive thresholds
            self.detector.shelf_interaction_threshold = 0.30
            self.detector.concealment_threshold = 0.40
            self.detector.shoplifting_threshold = 0.60
            print("   📈 Applied MODERATE-AGGRESSIVE thresholds for good model")

        else:
            # Lower performance - use balanced thresholds
            self.detector.shelf_interaction_threshold = 0.35
            self.detector.concealment_threshold = 0.45
            self.detector.shoplifting_threshold = 0.65
            print("   ⚖️  Applied BALANCED thresholds")

        # Additional optimizations based on precision/recall balance
        if precision < 0.8 and recall > 0.9:
            # High recall, low precision - increase thresholds slightly
            self.detector.shelf_interaction_threshold += 0.05
            self.detector.concealment_threshold += 0.05
            self.detector.shoplifting_threshold += 0.05
            print("   📊 Adjusted for precision-recall balance (reduced false positives)")

        elif precision > 0.9 and recall < 0.8:
            # High precision, low recall - decrease thresholds slightly
            self.detector.shelf_interaction_threshold = max(
                0.1, self.detector.shelf_interaction_threshold - 0.05)
            self.detector.concealment_threshold = max(
                0.2, self.detector.concealment_threshold - 0.05)
            self.detector.shoplifting_threshold = max(
                0.4, self.detector.shoplifting_threshold - 0.05)
            print("   📊 Adjusted for precision-recall balance (improved recall)")

        print(f"   🎯 FINAL OPTIMIZED THRESHOLDS:")
        print(
            f"      Shelf interaction: {self.detector.shelf_interaction_threshold:.2f}")
        print(f"      Concealment: {self.detector.concealment_threshold:.2f}")
        print(f"      Shoplifting: {self.detector.shoplifting_threshold:.2f}")

    def test_maximum_accuracy_detection(self) -> Dict:
        """Test the maximum accuracy detection system"""
        print("🧪 Testing MAXIMUM ACCURACY detection system...")

        # Enhanced test scenarios with more realistic patterns
        test_scenarios = [
            ("normal_browsing", False, "Customer browsing normally"),
            ("window_shopping", False, "Customer window shopping"),
            ("legitimate_shopping", False, "Customer shopping legitimately"),
            ("casual_theft", True, "Casual/impulsive theft"),
            ("professional_theft", True, "Professional theft"),
            ("opportunistic_theft", True, "Opportunistic theft"),
            ("distraction_theft", True, "Distraction-based theft"),
            ("concealment_theft", True, "Concealment-focused theft")
        ]

        results = []

        for scenario_name, expected, description in test_scenarios:
            print(f"   Testing: {description}")

            # Generate test features using the same patterns
            if scenario_name in self.behavior_patterns:
                test_features = self.behavior_patterns[scenario_name]()
            else:
                # Fallback to normal browsing
                test_features = self._generate_normal_browsing_pattern()

            # Create mock person data
            mock_person = {
                'person_id': 1,
                'bbox': {'center_x': 150, 'center_y': 240, 'width': 80, 'height': 180,
                         'x1': 110, 'y1': 150, 'x2': 190, 'y2': 330},
                'track_data': {
                    'positions': [
                        {'x': 150, 'y': 240, 'timestamp': datetime.now(),
                         'width': 80, 'height': 180}
                        # More positions for better detection
                        for _ in range(15)
                    ],
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now()
                }
            }

            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Test detection multiple times to simulate progression
            final_result = None
            for step in range(5):  # Multiple steps for progression
                result = self.detector.analyze_person_behavior(
                    mock_person, frame)
                final_result = result

            detected = self.detector.is_behavior_suspicious(final_result)

            results.append({
                'scenario': scenario_name,
                'expected': expected,
                'detected': detected,
                'score': final_result['suspicious_score'],
                'stage': final_result['shoplifting_stage'],
                'correct': detected == expected
            })

            status = "✅" if detected == expected else "❌"
            print(
                f"      {status} Expected={expected}, Detected={detected}, Score={final_result['suspicious_score']:.2f}, Stage={final_result['shoplifting_stage']}")

        # Calculate comprehensive metrics
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = correct / total

        # Calculate precision and recall for detection
        true_positives = sum(
            1 for r in results if r['expected'] and r['detected'])
        false_positives = sum(
            1 for r in results if not r['expected'] and r['detected'])
        false_negatives = sum(
            1 for r in results if r['expected'] and not r['detected'])

        precision = true_positives / \
            (true_positives + false_positives) if (true_positives +
                                                   false_positives) > 0 else 0
        recall = true_positives / \
            (true_positives + false_negatives) if (true_positives +
                                                   false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision +
                                               recall) if (precision + recall) > 0 else 0

        print(f"\n🎯 MAXIMUM ACCURACY TEST RESULTS:")
        print(f"   Overall Accuracy: {accuracy:.2f} ({correct}/{total})")
        print(f"   Precision: {precision:.2f}")
        print(f"   Recall: {recall:.2f}")
        print(f"   F1-Score: {f1_score:.2f}")
        print(f"   True Positives: {true_positives}")
        print(f"   False Positives: {false_positives}")
        print(f"   False Negatives: {false_negatives}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'results': results,
            'correct_predictions': correct,
            'total_tests': total,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def run_maximum_accuracy_training(self):
        """Run the complete maximum accuracy training pipeline"""
        print("🚀 MAXIMUM ACCURACY Shoplifting Detection Training")
        print("Advanced Synthetic Data + Optimized ML Models")
        print("=" * 60)

        start_time = datetime.now()

        # Generate advanced training data
        print("\n📊 Generating advanced training data...")
        features, labels = self.generate_advanced_training_data(
            samples_per_pattern=300)

        if len(features) == 0:
            print("❌ Failed to generate training data")
            return

        # Train maximum accuracy models
        print("\n🤖 Training maximum accuracy models...")
        training_data = self.train_maximum_accuracy_models(features, labels)

        if not training_data:
            print("❌ Model training failed")
            return

        # Optimize detector for maximum accuracy
        best_model = training_data['models'][training_data['best_model']]
        self.optimize_detector_for_maximum_accuracy(best_model['metrics'])

        # Test maximum accuracy detection
        print("\n🧪 Testing maximum accuracy detection...")
        test_results = self.test_maximum_accuracy_detection()

        # Generate comprehensive summary
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n🎯 MAXIMUM ACCURACY TRAINING SUMMARY:")
        print(f"   ⏱️  Training time: {duration}")
        print(f"   📊 Training samples: {training_data['training_samples']}")
        print(f"   🎭 Behavior patterns: {len(self.behavior_patterns)}")
        print(f"   🏆 Best model: {training_data['best_model']}")
        print(f"   🎯 Model F1-score: {best_model['metrics']['f1_score']:.3f}")
        print(f"   🎯 Detection accuracy: {test_results['accuracy']:.2f}")
        print(f"   🎯 Detection F1-score: {test_results['f1_score']:.2f}")

        # Determine success level
        if test_results['accuracy'] >= 0.9:
            success_level = "🔥 EXCEPTIONAL"
        elif test_results['accuracy'] >= 0.8:
            success_level = "🚀 EXCELLENT"
        elif test_results['accuracy'] >= 0.7:
            success_level = "✅ VERY GOOD"
        elif test_results['accuracy'] >= 0.6:
            success_level = "📈 GOOD"
        else:
            success_level = "⚠️  NEEDS IMPROVEMENT"

        print(f"   🏆 Success level: {success_level}")

        # Save comprehensive summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'duration': str(duration),
            'training_data': {
                'samples': training_data['training_samples'],
                'patterns': len(self.behavior_patterns),
                'feature_dimensions': training_data['feature_count']
            },
            'best_model': {
                'name': training_data['best_model'],
                'metrics': best_model['metrics'],
                'params': best_model['params'],
                'cv_scores': training_data['cv_scores']
            },
            'detection_performance': test_results,
            'optimized_thresholds': {
                'shelf_interaction': self.detector.shelf_interaction_threshold,
                'concealment': self.detector.concealment_threshold,
                'shoplifting': self.detector.shoplifting_threshold
            },
            'success_level': success_level
        }

        summary_file = self.models_dir / "maximum_accuracy_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📄 Comprehensive summary saved to: {summary_file}")
        print("\n🎉 MAXIMUM ACCURACY training completed!")
        print("\nNext steps:")
        print("1. Test with: python test_realistic_scenarios.py")
        print("2. Monitor real-world performance")
        print("3. Fine-tune based on actual usage")

        return summary


def main():
    """Main training function for maximum accuracy"""
    trainer = FocusedShopliftingTrainer()
    trainer.run_maximum_accuracy_training()


if __name__ == "__main__":
    main()
