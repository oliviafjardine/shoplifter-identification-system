#!/usr/bin/env python3
"""
Dataset-Only Model Training Script
Trains the shoplifting detection model using only the Kaggle dataset
"""

import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import kagglehub
from pathlib import Path
import time

def download_dataset():
    """Download the Kaggle dataset"""
    print("📥 Downloading Kaggle dataset...")
    try:
        # Download the dataset
        path = kagglehub.dataset_download("minhajuddinmeraj/anomalydetectiondatasetucf")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Find the shoplifting folder
        shoplifting_path = None
        for root, dirs, files in os.walk(path):
            if 'Shoplifting' in root or 'shoplifting' in root.lower():
                shoplifting_path = root
                break
        
        if not shoplifting_path:
            # Look for the specific folder mentioned in requirements
            anomaly_path = os.path.join(path, 'Anomaly-Videos-Part-4', 'Anomaly-Videos-Part-4Shoplifting')
            if os.path.exists(anomaly_path):
                shoplifting_path = anomaly_path
        
        if shoplifting_path and os.path.exists(shoplifting_path):
            print(f"✅ Found shoplifting videos at: {shoplifting_path}")
            return shoplifting_path
        else:
            print("⚠️ Shoplifting folder not found in dataset")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def extract_features_from_video(video_path):
    """Extract features from a video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        features = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames from the video
        sample_interval = max(1, total_frames // 30)  # Sample ~30 frames per video
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_interval == 0:
                # Extract basic features from frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Motion and activity features
                motion_intensity = np.std(gray)
                brightness = np.mean(gray)
                contrast = np.std(gray)
                
                # Edge detection for activity level
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                # Histogram features
                hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
                hist_features = hist.flatten() / np.sum(hist)
                
                # Combine features
                frame_features = [
                    motion_intensity,
                    brightness,
                    contrast,
                    edge_density,
                    np.var(gray),  # Texture variance
                    np.mean(np.abs(np.diff(gray, axis=0))),  # Vertical gradient
                    np.mean(np.abs(np.diff(gray, axis=1))),  # Horizontal gradient
                ]
                
                # Add histogram features
                frame_features.extend(hist_features[:8])  # Use first 8 histogram bins
                
                features.append(frame_features)
            
            frame_count += 1
        
        cap.release()
        
        if features:
            # Aggregate features across frames
            features_array = np.array(features)
            aggregated_features = [
                np.mean(features_array, axis=0),  # Mean across frames
                np.std(features_array, axis=0),   # Std across frames
                np.max(features_array, axis=0),   # Max across frames
                np.min(features_array, axis=0),   # Min across frames
            ]
            
            return np.concatenate(aggregated_features)
        
        return None
        
    except Exception as e:
        print(f"❌ Error processing video {video_path}: {e}")
        return None

def prepare_training_data(dataset_path):
    """Prepare training data from the dataset"""
    print("🔄 Preparing training data...")
    
    X = []
    y = []
    
    # Process shoplifting videos (positive examples)
    video_files = []
    for ext in ['*.avi', '*.mp4', '*.mov', '*.mkv']:
        video_files.extend(Path(dataset_path).glob(f"**/{ext}"))
    
    print(f"📹 Found {len(video_files)} video files")
    
    processed_count = 0
    for video_file in video_files:
        print(f"Processing: {video_file.name}")
        features = extract_features_from_video(str(video_file))
        
        if features is not None:
            X.append(features)
            y.append(1)  # Shoplifting = 1
            processed_count += 1
            
            if processed_count >= 50:  # Limit for demo
                break
    
    # Generate negative examples (normal behavior)
    print("🔄 Generating negative examples...")
    for i in range(len(X)):
        # Create synthetic normal behavior by modifying features
        normal_features = X[i].copy()
        
        # Reduce motion intensity and edge density for normal behavior
        normal_features[0] *= 0.5  # Lower motion
        normal_features[3] *= 0.7  # Lower edge density
        normal_features[4] *= 0.6  # Lower texture variance
        
        # Add some noise
        noise = np.random.normal(0, 0.1, len(normal_features))
        normal_features += noise
        
        X.append(normal_features)
        y.append(0)  # Normal = 0
    
    return np.array(X), np.array(y)

def train_model(X, y):
    """Train the shoplifting detection model"""
    print("🤖 Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Accuracy: {accuracy:.3f}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Shoplifting']))
    
    return model, accuracy

def save_model(model, accuracy):
    """Save the trained model"""
    models_dir = Path("ml/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'accuracy': accuracy,
        'timestamp': time.time(),
        'version': '3.0.0'
    }
    
    model_path = models_dir / "continuous_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 Model saved to: {model_path}")
    return model_path

def main():
    """Main training function"""
    print("🛡️ Starting Shoplifting Detection Model Training")
    print("=" * 60)
    
    # Download dataset
    dataset_path = download_dataset()
    if not dataset_path:
        print("❌ Failed to download dataset. Exiting.")
        return
    
    # Prepare training data
    X, y = prepare_training_data(dataset_path)
    
    if len(X) == 0:
        print("❌ No training data prepared. Exiting.")
        return
    
    print(f"📊 Training data prepared: {len(X)} samples")
    print(f"   - Shoplifting examples: {np.sum(y == 1)}")
    print(f"   - Normal examples: {np.sum(y == 0)}")
    
    # Train model
    model, accuracy = train_model(X, y)
    
    # Save model
    model_path = save_model(model, accuracy)
    
    print("\n🎉 Training completed successfully!")
    print(f"📈 Final accuracy: {accuracy:.3f}")
    print(f"💾 Model saved to: {model_path}")
    print("\n🚀 You can now run the main system with:")
    print("   python main.py")

if __name__ == "__main__":
    main()
