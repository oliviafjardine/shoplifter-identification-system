#!/usr/bin/env python3
"""
Focused Shoplifting Training System
Uses ONLY the shoplifting videos from UCF dataset to maximize detection accuracy
"""

from detection.shoplifting_detector import ShopliftingDetector
import os
import sys
import cv2
import numpy as np
import kagglehub
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))


class ShopliftingOnlyTrainer:
    """
    Focused trainer using ONLY shoplifting videos to maximize detection accuracy
    """

    def __init__(self):
        self.base_dir = Path("shoplifting_training")
        self.base_dir.mkdir(exist_ok=True)

        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.detector = ShopliftingDetector()
        self.scaler = StandardScaler()

        # Enhanced feature extraction
        self.feature_extractors = {
            'movement': self._extract_movement_features,
            'position': self._extract_position_features,
            'interaction': self._extract_interaction_features,
            'temporal': self._extract_temporal_features,
            'visual': self._extract_visual_features
        }

    def download_shoplifting_dataset(self) -> Optional[str]:
        """Download only the shoplifting folder from UCF dataset"""
        print("📥 Checking for existing shoplifting dataset...")

        # First check if we already have the data
        local_videos_dir = self.data_dir / "shoplifting_videos"
        if local_videos_dir.exists() and list(local_videos_dir.glob("*.mp4")):
            print(
                f"✅ Found existing shoplifting videos at: {local_videos_dir}")
            video_count = len(list(local_videos_dir.glob("*.mp4")))
            print(f"📹 Using {video_count} existing shoplifting videos")
            return str(local_videos_dir)

        print("📥 Downloading UCF Anomaly Detection dataset (shoplifting only)...")
        print("⚠️  This may take some time for the large dataset...")

        try:
            # Download the dataset
            dataset_path = kagglehub.dataset_download(
                "minhajuddinmeraj/anomalydetectiondatasetucf")
            dataset_path = Path(dataset_path)

            # Find shoplifting folder
            shoplifting_folder = dataset_path / "Anomaly-Videos-Part-4" / \
                "Anomaly-Videos-Part-4Shoplifting"

            if shoplifting_folder.exists():
                print(f"✅ Found shoplifting videos at: {shoplifting_folder}")

                # Count videos
                video_files = list(shoplifting_folder.glob("*.mp4"))
                print(f"📹 Found {len(video_files)} shoplifting videos")

                # Copy to our training directory for focused processing
                local_videos_dir.mkdir(exist_ok=True)

                print("📋 Copying shoplifting videos for focused training...")
                # Start with first 5 videos to avoid memory issues
                for i, video_file in enumerate(video_files[:5]):
                    local_path = local_videos_dir / video_file.name
                    if not local_path.exists():
                        import shutil
                        shutil.copy2(video_file, local_path)
                        print(f"   Copied: {video_file.name} ({i+1}/5)")

                return str(local_videos_dir)
            else:
                print("❌ Shoplifting folder not found!")
                print("Available folders:")
                for folder in dataset_path.rglob("*"):
                    if folder.is_dir() and "shoplifting" in folder.name.lower():
                        print(f"  - {folder}")
                return None

        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print(
                "💡 You can manually place shoplifting videos in: shoplifting_training/data/shoplifting_videos/")
            return None

    def extract_shoplifting_features(self, videos_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract comprehensive features from shoplifting videos"""
        print("🎬 Extracting features from shoplifting videos...")

        videos_path = Path(videos_dir)
        video_files = list(videos_path.glob("*.mp4"))

        all_features = []
        all_labels = []

        for i, video_file in enumerate(video_files):
            print(
                f"   Processing {video_file.name} ({i+1}/{len(video_files)})...")

            try:
                cap = cv2.VideoCapture(str(video_file))
                if not cap.isOpened():
                    continue

                # Extract features from video segments
                video_features = self._extract_video_features(cap)

                # Create positive and negative samples from the same video
                # Early frames = normal behavior (label 0)
                # Later frames = shoplifting behavior (label 1)

                total_features = len(video_features)
                if total_features < 10:
                    continue

                # First 30% = normal behavior
                normal_end = int(total_features * 0.3)
                # Last 50% = shoplifting behavior
                shoplifting_start = int(total_features * 0.5)

                # Add normal behavior samples
                for j in range(0, normal_end):
                    all_features.append(video_features[j])
                    all_labels.append(0)  # Normal

                # Add shoplifting behavior samples
                for j in range(shoplifting_start, total_features):
                    all_features.append(video_features[j])
                    all_labels.append(1)  # Shoplifting

                cap.release()

            except Exception as e:
                print(f"      Error processing {video_file.name}: {e}")
                continue

        features_array = np.array(all_features)
        labels_array = np.array(all_labels)

        print(f"✅ Extracted features:")
        print(f"   Total samples: {len(features_array)}")
        print(
            f"   Feature dimensions: {features_array.shape[1] if len(features_array) > 0 else 0}")
        print(f"   Normal samples: {np.sum(labels_array == 0)}")
        print(f"   Shoplifting samples: {np.sum(labels_array == 1)}")

        return features_array, labels_array

    def _extract_video_features(self, cap) -> List[np.ndarray]:
        """Extract features from video frames"""
        features = []
        frame_count = 0
        prev_frame = None

        # Sample every 30 frames (about 1 second at 30fps)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 30 == 0:  # Sample every 30 frames
                try:
                    frame_features = self._extract_frame_features(
                        frame, prev_frame)
                    if frame_features is not None:
                        features.append(frame_features)
                    prev_frame = frame.copy()
                except Exception as e:
                    continue

            frame_count += 1

        return features

    def _extract_frame_features(self, frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Extract comprehensive features from a single frame"""
        try:
            features = []

            # Basic frame properties
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Movement features
            movement_features = self._extract_movement_features(
                frame, prev_frame)
            features.extend(movement_features)

            # 2. Position features
            position_features = self._extract_position_features(frame)
            features.extend(position_features)

            # 3. Interaction features
            interaction_features = self._extract_interaction_features(frame)
            features.extend(interaction_features)

            # 4. Temporal features
            temporal_features = self._extract_temporal_features(frame)
            features.extend(temporal_features)

            # 5. Visual features
            visual_features = self._extract_visual_features(frame)
            features.extend(visual_features)

            return np.array(features)

        except Exception as e:
            return None

    def _extract_movement_features(self, frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> List[float]:
        """Extract movement-related features"""
        features = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            # Optical flow
            flow = cv2.calcOpticalFlowPyrLK(prev_gray, gray, None, None)
            if flow[0] is not None:
                # Movement magnitude
                movement_mag = np.mean(
                    np.sqrt(flow[0][:, :, 0]**2 + flow[0][:, :, 1]**2))
                features.append(movement_mag)

                # Movement direction variance
                angles = np.arctan2(flow[0][:, :, 1], flow[0][:, :, 0])
                direction_var = np.var(angles)
                features.append(direction_var)
            else:
                features.extend([0.0, 0.0])

            # Frame difference
            diff = cv2.absdiff(gray, prev_gray)
            motion_energy = np.mean(diff)
            features.append(motion_energy)
        else:
            features.extend([0.0, 0.0, 0.0])

        return features

    def _extract_position_features(self, frame: np.ndarray) -> List[float]:
        """Extract position-related features"""
        features = []

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect potential person regions using simple blob detection
        # This is a simplified approach - in production you'd use proper person detection

        # Edge density in different regions
        edges = cv2.Canny(gray, 50, 150)

        # Divide frame into regions
        h_third = height // 3
        w_third = width // 3

        regions = [
            edges[0:h_third, 0:w_third],           # Top-left
            edges[0:h_third, w_third:2*w_third],   # Top-center
            edges[0:h_third, 2*w_third:width],     # Top-right
            edges[h_third:2*h_third, 0:w_third],   # Mid-left
            edges[h_third:2*h_third, w_third:2*w_third],  # Mid-center
            edges[h_third:2*h_third, 2*w_third:width],   # Mid-right
            edges[2*h_third:height, 0:w_third],    # Bottom-left
            edges[2*h_third:height, w_third:2*w_third],  # Bottom-center
            edges[2*h_third:height, 2*w_third:width]    # Bottom-right
        ]

        # Edge density in each region
        for region in regions:
            if region.size > 0:
                edge_density = np.sum(region > 0) / region.size
                features.append(edge_density)
            else:
                features.append(0.0)

        return features

    def _extract_interaction_features(self, frame: np.ndarray) -> List[float]:
        """Extract interaction-related features"""
        features = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Texture analysis
        # Local Binary Pattern approximation
        texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(texture_var)

        # Contour analysis
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Number of contours
            features.append(len(contours))

            # Average contour area
            areas = [cv2.contourArea(c) for c in contours]
            avg_area = np.mean(areas) if areas else 0
            features.append(avg_area)

            # Contour complexity
            perimeters = [cv2.arcLength(c, True) for c in contours]
            avg_perimeter = np.mean(perimeters) if perimeters else 0
            features.append(avg_perimeter)
        else:
            features.extend([0.0, 0.0, 0.0])

        return features

    def _extract_temporal_features(self, frame: np.ndarray) -> List[float]:
        """Extract temporal-related features"""
        features = []

        # Brightness statistics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(np.min(gray))
        features.append(np.max(gray))

        return features

    def _extract_visual_features(self, frame: np.ndarray) -> List[float]:
        """Extract visual appearance features"""
        features = []

        # Color statistics
        for channel in range(3):  # BGR channels
            channel_data = frame[:, :, channel]
            features.append(np.mean(channel_data))
            features.append(np.std(channel_data))

        # Histogram features
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist_features = hist.flatten() / np.sum(hist)  # Normalize
        features.extend(hist_features)

        return features

    def train_optimized_models(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Train multiple optimized models for maximum accuracy"""
        print("🤖 Training optimized models for maximum accuracy...")

        if len(features) == 0:
            print("❌ No training data available")
            return {}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {}

        # 1. Random Forest with hyperparameter tuning
        print("   Training Random Forest...")
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced']
        }
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1)
        rf_grid.fit(X_train_scaled, y_train)

        rf_pred = rf_grid.predict(X_test_scaled)
        rf_metrics = self._calculate_metrics(y_test, rf_pred)
        models['random_forest'] = {
            'model': rf_grid.best_estimator_,
            'metrics': rf_metrics,
            'params': rf_grid.best_params_
        }

        # 2. Gradient Boosting
        print("   Training Gradient Boosting...")
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        gb = GradientBoostingClassifier(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='f1', n_jobs=-1)
        gb_grid.fit(X_train_scaled, y_train)

        gb_pred = gb_grid.predict(X_test_scaled)
        gb_metrics = self._calculate_metrics(y_test, gb_pred)
        models['gradient_boosting'] = {
            'model': gb_grid.best_estimator_,
            'metrics': gb_metrics,
            'params': gb_grid.best_params_
        }

        # 3. SVM
        print("   Training SVM...")
        svm_params = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'class_weight': ['balanced']
        }
        svm = SVC(random_state=42, probability=True)
        svm_grid = GridSearchCV(svm, svm_params, cv=3, scoring='f1', n_jobs=-1)
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

        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Accuracy: {best_model['metrics']['accuracy']:.3f}")
        print(f"   Precision: {best_model['metrics']['precision']:.3f}")
        print(f"   Recall: {best_model['metrics']['recall']:.3f}")
        print(f"   F1-Score: {best_model['metrics']['f1_score']:.3f}")

        # Save models
        training_data = {
            'models': models,
            'best_model': best_model_name,
            'scaler': self.scaler,
            'feature_count': features.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

        models_file = self.models_dir / "shoplifting_optimized_models.pkl"
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

    def optimize_detector_thresholds(self, best_model_metrics: Dict):
        """Optimize detector thresholds based on best model performance"""
        print("⚙️  Optimizing detector thresholds for maximum accuracy...")

        accuracy = best_model_metrics['accuracy']
        precision = best_model_metrics['precision']
        recall = best_model_metrics['recall']
        f1 = best_model_metrics['f1_score']

        # Aggressive optimization for high accuracy
        if accuracy > 0.9 and f1 > 0.9:
            # Excellent performance - use aggressive thresholds
            self.detector.shelf_interaction_threshold = 0.25
            self.detector.concealment_threshold = 0.35
            self.detector.shoplifting_threshold = 0.55
            print("   🎯 Applied aggressive thresholds for high-accuracy model")

        elif accuracy > 0.8 and f1 > 0.8:
            # Good performance - use moderate thresholds
            self.detector.shelf_interaction_threshold = 0.30
            self.detector.concealment_threshold = 0.40
            self.detector.shoplifting_threshold = 0.60
            print("   📈 Applied moderate thresholds for good model")

        else:
            # Lower performance - use conservative thresholds
            self.detector.shelf_interaction_threshold = 0.35
            self.detector.concealment_threshold = 0.45
            self.detector.shoplifting_threshold = 0.65
            print("   📊 Applied conservative thresholds")

        print(f"   New thresholds:")
        print(
            f"     Shelf interaction: {self.detector.shelf_interaction_threshold:.2f}")
        print(f"     Concealment: {self.detector.concealment_threshold:.2f}")
        print(f"     Shoplifting: {self.detector.shoplifting_threshold:.2f}")

    def run_complete_training(self):
        """Run the complete focused training pipeline"""
        print("🎯 Shoplifting-Only Training Pipeline")
        print("Using UCF Anomaly Detection - Shoplifting Videos ONLY")
        print("=" * 60)

        start_time = datetime.now()

        # Download shoplifting dataset
        videos_dir = self.download_shoplifting_dataset()
        if not videos_dir:
            print("❌ Failed to download shoplifting dataset")
            return

        # Extract features from shoplifting videos
        features, labels = self.extract_shoplifting_features(videos_dir)
        if len(features) == 0:
            print("❌ No features extracted from videos")
            return

        # Train optimized models
        training_data = self.train_optimized_models(features, labels)
        if not training_data:
            print("❌ Model training failed")
            return

        # Optimize detector thresholds
        best_model = training_data['models'][training_data['best_model']]
        self.optimize_detector_thresholds(best_model['metrics'])

        # Generate summary
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n🎯 Training Summary:")
        print(f"   Training time: {duration}")
        print(
            f"   Videos processed: {len(list(Path(videos_dir).glob('*.mp4')))}")
        print(f"   Features extracted: {len(features)}")
        print(f"   Feature dimensions: {features.shape[1]}")
        print(f"   Best model: {training_data['best_model']}")
        print(f"   Best accuracy: {best_model['metrics']['accuracy']:.3f}")
        print(f"   Best F1-score: {best_model['metrics']['f1_score']:.3f}")

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'duration': str(duration),
            'videos_processed': len(list(Path(videos_dir).glob('*.mp4'))),
            'features_extracted': len(features),
            'feature_dimensions': features.shape[1],
            'best_model': training_data['best_model'],
            'best_metrics': best_model['metrics'],
            'optimized_thresholds': {
                'shelf_interaction': self.detector.shelf_interaction_threshold,
                'concealment': self.detector.concealment_threshold,
                'shoplifting': self.detector.shoplifting_threshold
            }
        }

        summary_file = self.models_dir / "shoplifting_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📄 Training summary saved to: {summary_file}")
        print("\n✅ Shoplifting-focused training completed!")
        print("\nNext steps:")
        print("1. Test improved detection: python test_realistic_scenarios.py")
        print("2. Monitor performance in real scenarios")
        print("3. Retrain with more videos if needed")


def main():
    """Main training function"""
    trainer = ShopliftingOnlyTrainer()
    trainer.run_complete_training()


if __name__ == "__main__":
    main()
