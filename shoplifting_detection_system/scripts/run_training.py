#!/usr/bin/env python3
"""
Complete Training Pipeline with Kaggle Dataset
Downloads UCF Anomaly Detection dataset and trains the shoplifting detection model
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add training directory to path
sys.path.append(str(Path(__file__).parent / "training"))

from training.dataset_manager import ShopliftingDatasetManager
from training.model_trainer import ShopliftingModelTrainer
from training.evaluate_model import ModelEvaluator

def main():
    """Complete training pipeline"""
    print("🚀 Shoplifting Detection Training Pipeline")
    print("Using UCF Anomaly Detection Dataset - Shoplifting Videos")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Step 1: Download and prepare dataset
    print("\n📥 STEP 1: Dataset Preparation")
    print("-" * 30)
    
    dataset_manager = ShopliftingDatasetManager()
    
    # Download dataset
    dataset_path = dataset_manager.download_dataset()
    if not dataset_path:
        print("❌ Failed to download dataset. Exiting.")
        return
    
    # Analyze videos
    print("\n🔍 Analyzing videos...")
    analysis = dataset_manager.analyze_videos()
    
    if analysis['total_videos'] == 0:
        print("❌ No videos found in dataset. Exiting.")
        return
    
    print(f"✅ Found {analysis['total_videos']} shoplifting videos")
    print(f"   Total duration: {analysis['summary']['total_duration']:.1f} seconds")
    print(f"   Average FPS: {analysis['summary']['avg_fps']:.1f}")
    
    # Extract frames
    print("\n🎬 Extracting frames for training...")
    frames_path = dataset_manager.extract_frames(
        max_videos=5,  # Start with 5 videos for initial training
        frame_interval=90  # Every 90 frames (about 3 seconds at 30fps)
    )
    
    if not frames_path:
        print("❌ Failed to extract frames. Exiting.")
        return
    
    # Create annotations
    print("\n📝 Creating training annotations...")
    annotations_path = dataset_manager.create_training_annotations()
    
    if not annotations_path:
        print("❌ Failed to create annotations. Exiting.")
        return
    
    # Step 2: Train the model
    print("\n🤖 STEP 2: Model Training")
    print("-" * 30)
    
    trainer = ShopliftingModelTrainer()
    
    # Run complete training
    training_summary = trainer.train_complete_system()
    
    if not training_summary:
        print("❌ Training failed. Exiting.")
        return
    
    print("✅ Model training completed!")
    print(f"   Training samples: {training_summary['training_data']['total_samples']}")
    print(f"   Positive samples: {training_summary['training_data']['positive_samples']}")
    print(f"   Feature dimensions: {training_summary['training_data']['feature_dimensions']}")
    
    # Step 3: Evaluate the model
    print("\n📊 STEP 3: Model Evaluation")
    print("-" * 30)
    
    evaluator = ModelEvaluator()
    
    # Load trained models
    if not evaluator.load_trained_models():
        print("❌ Failed to load trained models for evaluation.")
        return
    
    # Run evaluation
    evaluation_results = evaluator.evaluate_on_test_data()
    improvement_results = evaluator.test_detection_improvement()
    
    # Generate comprehensive report
    report_file = evaluator.generate_comprehensive_report()
    
    # Step 4: Summary and next steps
    print("\n🎯 STEP 4: Training Summary")
    print("-" * 30)
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print(f"⏱️  Total training time: {training_duration}")
    print(f"📊 Dataset summary:")
    print(f"   Videos processed: {analysis['total_videos']}")
    print(f"   Frames extracted: {training_summary['training_data']['total_samples']}")
    print(f"   Shoplifting samples: {training_summary['training_data']['positive_samples']}")
    
    if evaluation_results:
        rf_performance = evaluation_results.get('random_forest', {})
        print(f"\n🤖 Model performance:")
        print(f"   Accuracy: {rf_performance.get('accuracy', 0):.3f}")
        print(f"   Precision: {rf_performance.get('precision', 0):.3f}")
        print(f"   Recall: {rf_performance.get('recall', 0):.3f}")
        print(f"   F1-Score: {rf_performance.get('f1_score', 0):.3f}")
    
    if improvement_results:
        improvement = improvement_results.get('improvement', {})
        print(f"\n📈 Detection improvement:")
        print(f"   Before training: {improvement.get('accuracy_before', 0):.2f}")
        print(f"   After training: {improvement.get('accuracy_after', 0):.2f}")
        print(f"   Improvement: {improvement.get('improvement', 0):.2f}")
    
    print(f"\n📄 Detailed report: {report_file}")
    
    # Recommendations
    print("\n🎯 Next Steps:")
    print("1. Review the evaluation report and plots")
    print("2. Test the improved system:")
    print("   python test_realistic_scenarios.py")
    print("3. Monitor performance in real-world scenarios")
    print("4. Retrain with more data as needed")
    
    # Check if training was successful
    if evaluation_results and evaluation_results.get('random_forest', {}).get('accuracy', 0) > 0.7:
        print("\n✅ Training successful! Model shows good performance.")
    elif improvement_results and improvement_results.get('improvement', {}).get('improvement', 0) > 0:
        print("\n✅ Training successful! Detection system improved.")
    else:
        print("\n⚠️  Training completed but performance may need improvement.")
        print("   Consider:")
        print("   - Training with more videos")
        print("   - Adjusting feature extraction")
        print("   - Fine-tuning model parameters")
    
    print("\n🎉 Training pipeline completed!")

def quick_test():
    """Quick test to verify the training worked"""
    print("\n🧪 Quick Test of Trained Model")
    print("-" * 30)
    
    try:
        from detection.shoplifting_detector import ShopliftingDetector
        import numpy as np
        
        detector = ShopliftingDetector()
        
        # Create test scenario
        mock_person_data = {
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
        result = detector.analyze_person_behavior(mock_person_data, frame)
        
        print(f"✅ Quick test successful!")
        print(f"   Detection stage: {result['shoplifting_stage']}")
        print(f"   Suspicious score: {result['suspicious_score']:.2f}")
        print(f"   Behaviors detected: {len(result['behaviors'])}")
        
        # Show current thresholds
        print(f"\n⚙️  Current detection thresholds:")
        print(f"   Shelf interaction: {detector.shelf_interaction_threshold:.2f}")
        print(f"   Concealment: {detector.concealment_threshold:.2f}")
        print(f"   Shoplifting: {detector.shoplifting_threshold:.2f}")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

if __name__ == "__main__":
    try:
        main()
        quick_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("Please check the error and try again.")
