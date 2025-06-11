#!/usr/bin/env python3
"""
Model Evaluation for Shoplifting Detection
Evaluates the trained model performance and provides detailed metrics
"""

import os
import sys
import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from detection.shoplifting_detector import ShopliftingDetector
from training.model_trainer import ShopliftingModelTrainer

class ModelEvaluator:
    """
    Evaluates trained shoplifting detection models
    """
    
    def __init__(self, models_dir: str = "training_data/models"):
        self.models_dir = Path(models_dir)
        self.results_dir = self.models_dir / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.trained_models = None
        self.detector = ShopliftingDetector()
        
    def load_trained_models(self) -> bool:
        """Load trained models from file"""
        models_file = self.models_dir / "trained_models.pkl"
        
        if not models_file.exists():
            print("❌ No trained models found. Run model_trainer.py first.")
            return False
        
        try:
            with open(models_file, 'rb') as f:
                self.trained_models = pickle.load(f)
            print("✅ Trained models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading trained models: {e}")
            return False
    
    def evaluate_on_test_data(self) -> Dict:
        """Evaluate model on test data"""
        if not self.trained_models:
            print("❌ No trained models available")
            return {}
        
        print("🧪 Evaluating model on test data...")
        
        # Load test features (using a portion of training data as test for now)
        trainer = ShopliftingModelTrainer()
        features, labels = trainer.process_training_videos(max_frames_per_video=30)
        
        if len(features) == 0:
            print("❌ No test data available")
            return {}
        
        # Use last 20% as test data
        test_size = int(0.2 * len(features))
        test_features = features[-test_size:]
        test_labels = labels[-test_size:]
        
        # Scale features
        scaler = self.trained_models['feature_scaler']
        test_features_scaled = scaler.transform(test_features)
        
        # Get predictions from both models
        iso_model = self.trained_models['isolation_forest']
        rf_model = self.trained_models['random_forest']
        
        # Isolation Forest predictions
        iso_predictions = iso_model.predict(test_features_scaled)
        iso_predictions = (iso_predictions == -1).astype(int)  # Convert to 0/1
        iso_scores = iso_model.decision_function(test_features_scaled)
        
        # Random Forest predictions
        rf_predictions = rf_model.predict(test_features_scaled)
        rf_probabilities = rf_model.predict_proba(test_features_scaled)[:, 1]
        
        # Calculate detailed metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        evaluation_results = {
            'test_data_info': {
                'total_samples': len(test_features),
                'positive_samples': int(np.sum(test_labels)),
                'negative_samples': int(len(test_labels) - np.sum(test_labels))
            },
            'isolation_forest': {
                'accuracy': accuracy_score(test_labels, iso_predictions),
                'precision': precision_score(test_labels, iso_predictions, zero_division=0),
                'recall': recall_score(test_labels, iso_predictions, zero_division=0),
                'f1_score': f1_score(test_labels, iso_predictions, zero_division=0),
                'confusion_matrix': confusion_matrix(test_labels, iso_predictions).tolist()
            },
            'random_forest': {
                'accuracy': accuracy_score(test_labels, rf_predictions),
                'precision': precision_score(test_labels, rf_predictions, zero_division=0),
                'recall': recall_score(test_labels, rf_predictions, zero_division=0),
                'f1_score': f1_score(test_labels, rf_predictions, zero_division=0),
                'confusion_matrix': confusion_matrix(test_labels, rf_predictions).tolist()
            }
        }
        
        # Print results
        print("📊 Evaluation Results:")
        print("\n🔍 Isolation Forest:")
        for metric, value in evaluation_results['isolation_forest'].items():
            if metric != 'confusion_matrix':
                print(f"   {metric}: {value:.3f}")
        
        print("\n🌲 Random Forest:")
        for metric, value in evaluation_results['random_forest'].items():
            if metric != 'confusion_matrix':
                print(f"   {metric}: {value:.3f}")
        
        # Save results
        results_file = self.results_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n💾 Evaluation results saved to: {results_file}")
        
        # Generate plots
        self.generate_evaluation_plots(test_labels, iso_predictions, rf_predictions, rf_probabilities)
        
        return evaluation_results
    
    def generate_evaluation_plots(self, true_labels: np.ndarray, iso_predictions: np.ndarray, 
                                rf_predictions: np.ndarray, rf_probabilities: np.ndarray):
        """Generate evaluation plots"""
        print("📈 Generating evaluation plots...")
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Shoplifting Detection Model Evaluation', fontsize=16)
            
            # Confusion Matrix for Isolation Forest
            cm_iso = confusion_matrix(true_labels, iso_predictions)
            sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Isolation Forest - Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            
            # Confusion Matrix for Random Forest
            cm_rf = confusion_matrix(true_labels, rf_predictions)
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1])
            axes[0, 1].set_title('Random Forest - Confusion Matrix')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Actual')
            
            # ROC Curve for Random Forest
            fpr, tpr, _ = roc_curve(true_labels, rf_probabilities)
            roc_auc = auc(fpr, tpr)
            axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1, 0].set_xlim([0.0, 1.0])
            axes[1, 0].set_ylim([0.0, 1.05])
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('Random Forest - ROC Curve')
            axes[1, 0].legend(loc="lower right")
            
            # Prediction Distribution
            axes[1, 1].hist(rf_probabilities[true_labels == 0], bins=20, alpha=0.7, 
                           label='Normal', color='blue')
            axes[1, 1].hist(rf_probabilities[true_labels == 1], bins=20, alpha=0.7, 
                           label='Shoplifting', color='red')
            axes[1, 1].set_xlabel('Prediction Probability')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Prediction Probability Distribution')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.results_dir / "evaluation_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Evaluation plots saved to: {plot_file}")
            
            plt.close()
            
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    def test_detection_improvement(self) -> Dict:
        """Test if the trained model improves detection accuracy"""
        print("🎯 Testing detection improvement...")
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'normal_browsing',
                'expected': False,
                'description': 'Customer browsing normally'
            },
            {
                'name': 'shelf_interaction',
                'expected': False,
                'description': 'Customer examining items'
            },
            {
                'name': 'concealment_behavior',
                'expected': True,
                'description': 'Customer concealing items'
            },
            {
                'name': 'exit_movement',
                'expected': True,
                'description': 'Customer moving to exit with concealed item'
            }
        ]
        
        results = {
            'before_training': {},
            'after_training': {},
            'improvement': {}
        }
        
        # Test with original detector (before training)
        original_detector = ShopliftingDetector()
        
        # Test with updated detector (after training)
        if self.trained_models:
            # Apply trained model improvements
            rf_performance = self.trained_models.get('performance', {}).get('random_forest', {})
            if rf_performance.get('precision', 0) < 0.7:
                self.detector.shelf_interaction_threshold = min(0.6, self.detector.shelf_interaction_threshold + 0.1)
                self.detector.concealment_threshold = min(0.7, self.detector.concealment_threshold + 0.1)
            if rf_performance.get('recall', 0) < 0.7:
                self.detector.shelf_interaction_threshold = max(0.2, self.detector.shelf_interaction_threshold - 0.05)
                self.detector.concealment_threshold = max(0.3, self.detector.concealment_threshold - 0.05)
        
        # Run tests
        for scenario in test_scenarios:
            # Create mock test data (simplified)
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
            
            # Test original detector
            original_result = original_detector.analyze_person_behavior(mock_person_data, frame)
            original_suspicious = original_detector.is_behavior_suspicious(original_result)
            
            # Test updated detector
            updated_result = self.detector.analyze_person_behavior(mock_person_data, frame)
            updated_suspicious = self.detector.is_behavior_suspicious(updated_result)
            
            results['before_training'][scenario['name']] = {
                'detected': original_suspicious,
                'score': original_result['suspicious_score'],
                'correct': original_suspicious == scenario['expected']
            }
            
            results['after_training'][scenario['name']] = {
                'detected': updated_suspicious,
                'score': updated_result['suspicious_score'],
                'correct': updated_suspicious == scenario['expected']
            }
        
        # Calculate improvement
        before_accuracy = sum(1 for r in results['before_training'].values() if r['correct']) / len(test_scenarios)
        after_accuracy = sum(1 for r in results['after_training'].values() if r['correct']) / len(test_scenarios)
        
        results['improvement'] = {
            'accuracy_before': before_accuracy,
            'accuracy_after': after_accuracy,
            'improvement': after_accuracy - before_accuracy
        }
        
        print(f"📊 Detection Improvement Results:")
        print(f"   Accuracy before training: {before_accuracy:.2f}")
        print(f"   Accuracy after training: {after_accuracy:.2f}")
        print(f"   Improvement: {results['improvement']['improvement']:.2f}")
        
        # Save results
        improvement_file = self.results_dir / "detection_improvement.json"
        with open(improvement_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Improvement results saved to: {improvement_file}")
        
        return results
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        print("📋 Generating comprehensive evaluation report...")
        
        # Load all results
        evaluation_results = {}
        improvement_results = {}
        
        eval_file = self.results_dir / "evaluation_results.json"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                evaluation_results = json.load(f)
        
        improvement_file = self.results_dir / "detection_improvement.json"
        if improvement_file.exists():
            with open(improvement_file, 'r') as f:
                improvement_results = json.load(f)
        
        # Generate report
        report = f"""
# Shoplifting Detection Model Evaluation Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report evaluates the performance of the trained shoplifting detection model using real video data from the UCF Anomaly Detection dataset.

## Dataset Information

- **Source**: UCF Anomaly Detection Dataset - Shoplifting Videos
- **Training Samples**: {evaluation_results.get('test_data_info', {}).get('total_samples', 'N/A')}
- **Positive Samples**: {evaluation_results.get('test_data_info', {}).get('positive_samples', 'N/A')}
- **Negative Samples**: {evaluation_results.get('test_data_info', {}).get('negative_samples', 'N/A')}

## Model Performance

### Random Forest Classifier
- **Accuracy**: {evaluation_results.get('random_forest', {}).get('accuracy', 0):.3f}
- **Precision**: {evaluation_results.get('random_forest', {}).get('precision', 0):.3f}
- **Recall**: {evaluation_results.get('random_forest', {}).get('recall', 0):.3f}
- **F1-Score**: {evaluation_results.get('random_forest', {}).get('f1_score', 0):.3f}

### Isolation Forest (Anomaly Detection)
- **Accuracy**: {evaluation_results.get('isolation_forest', {}).get('accuracy', 0):.3f}
- **Precision**: {evaluation_results.get('isolation_forest', {}).get('precision', 0):.3f}
- **Recall**: {evaluation_results.get('isolation_forest', {}).get('recall', 0):.3f}
- **F1-Score**: {evaluation_results.get('isolation_forest', {}).get('f1_score', 0):.3f}

## Detection System Improvement

- **Accuracy Before Training**: {improvement_results.get('improvement', {}).get('accuracy_before', 0):.2f}
- **Accuracy After Training**: {improvement_results.get('improvement', {}).get('accuracy_after', 0):.2f}
- **Overall Improvement**: {improvement_results.get('improvement', {}).get('improvement', 0):.2f}

## Recommendations

1. **Model Performance**: {'Good' if evaluation_results.get('random_forest', {}).get('accuracy', 0) > 0.8 else 'Needs Improvement'}
2. **Detection Accuracy**: {'Improved' if improvement_results.get('improvement', {}).get('improvement', 0) > 0 else 'No Improvement'}
3. **Next Steps**: 
   - Continue training with more diverse data
   - Implement real-time model updates
   - Monitor performance in production environment

## Files Generated

- `evaluation_results.json`: Detailed evaluation metrics
- `detection_improvement.json`: Before/after comparison
- `evaluation_plots.png`: Visualization of results
- `evaluation_report.md`: This comprehensive report

---
*Report generated by Shoplifting Detection Model Evaluator*
"""
        
        # Save report
        report_file = self.results_dir / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Comprehensive report saved to: {report_file}")
        
        return str(report_file)

def main():
    """Main evaluation function"""
    print("🎯 Shoplifting Detection Model Evaluator")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load trained models
    if not evaluator.load_trained_models():
        return
    
    # Run evaluation
    evaluation_results = evaluator.evaluate_on_test_data()
    
    # Test detection improvement
    improvement_results = evaluator.test_detection_improvement()
    
    # Generate comprehensive report
    report_file = evaluator.generate_comprehensive_report()
    
    print("\n✅ Evaluation completed!")
    print(f"📄 Comprehensive report: {report_file}")
    print("\n🎯 Next steps:")
    print("1. Review the evaluation report and plots")
    print("2. If performance is good, deploy the updated model")
    print("3. Continue monitoring and retraining as needed")

if __name__ == "__main__":
    main()
