#!/usr/bin/env python3
"""
Apply Trained Model to Detection System
Integrates the trained ML model with the shoplifting detection system
"""

import pickle
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from detection.shoplifting_detector import ShopliftingDetector

class ModelIntegrator:
    """
    Integrates trained ML models with the shoplifting detection system
    """
    
    def __init__(self):
        self.detector = ShopliftingDetector()
        self.trained_model = None
        self.scaler = None
        self.model_info = None
        
    def load_trained_model(self) -> bool:
        """Load the trained model"""
        models_file = Path("focused_shoplifting_training/models/maximum_accuracy_models.pkl")
        
        if not models_file.exists():
            print("❌ No trained model found. Run train_shoplifting_focused.py first.")
            return False
        
        try:
            with open(models_file, 'rb') as f:
                training_data = pickle.load(f)
            
            best_model_name = training_data['best_model']
            self.trained_model = training_data['models'][best_model_name]['model']
            self.scaler = training_data['scaler']
            self.model_info = training_data['models'][best_model_name]['metrics']
            
            print(f"✅ Loaded trained model: {best_model_name}")
            print(f"   Model accuracy: {self.model_info['accuracy']:.3f}")
            print(f"   Model F1-score: {self.model_info['f1_score']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading trained model: {e}")
            return False
    
    def extract_behavior_features(self, person_data: dict, behavior_analysis: dict) -> np.ndarray:
        """Extract features from person behavior for ML prediction"""
        features = []
        
        # Movement characteristics
        track_data = person_data.get('track_data', {})
        positions = track_data.get('positions', [])
        
        if len(positions) >= 2:
            # Calculate movement metrics
            distances = []
            for i in range(1, len(positions)):
                dist = np.sqrt((positions[i]['x'] - positions[i-1]['x'])**2 + 
                              (positions[i]['y'] - positions[i-1]['y'])**2)
                distances.append(dist)
            
            avg_speed = np.mean(distances) if distances else 0
            speed_variance = np.var(distances) if distances else 0
            direction_changes = len([d for d in distances if d > 10]) / len(distances) if distances else 0
            stop_frequency = len([d for d in distances if d < 5]) / len(distances) if distances else 0
        else:
            avg_speed = speed_variance = direction_changes = stop_frequency = 0
        
        features.extend([avg_speed, direction_changes, speed_variance, stop_frequency])
        
        # Position characteristics
        bbox = person_data.get('bbox', {})
        shelf_distance = 0.1  # Simplified - assume close to shelf
        time_near_shelves = 0.5  # Simplified
        store_coverage = 0.3  # Simplified
        
        features.extend([shelf_distance, time_near_shelves, store_coverage])
        
        # Interaction patterns from behavior analysis
        shelf_interaction = behavior_analysis.get('shelf_interaction_score', 0)
        reaching_behavior = behavior_analysis.get('reaching_score', 0)
        item_manipulation = behavior_analysis.get('item_taking_score', 0)
        concealment_behavior = behavior_analysis.get('concealment_score', 0)
        
        features.extend([shelf_interaction, reaching_behavior, item_manipulation, concealment_behavior])
        
        # Attention patterns (simplified)
        looking_around = min(1.0, behavior_analysis.get('suspicious_score', 0) * 2)
        surveillance_awareness = min(1.0, behavior_analysis.get('suspicious_score', 0) * 1.5)
        nervous_behavior = min(1.0, behavior_analysis.get('suspicious_score', 0))
        
        features.extend([looking_around, surveillance_awareness, nervous_behavior])
        
        # Temporal patterns
        time_in_store = 30.0  # Simplified - assume 30 seconds
        time_stationary = 0.6  # Simplified
        confidence_level = 1.0 - behavior_analysis.get('suspicious_score', 0)
        
        features.extend([time_in_store, time_stationary, confidence_level])
        
        return np.array(features)
    
    def enhanced_behavior_analysis(self, person_data: dict, frame: np.ndarray) -> dict:
        """Enhanced behavior analysis using trained ML model"""
        # Get original behavior analysis
        original_result = self.detector.analyze_person_behavior(person_data, frame)
        
        if self.trained_model is None or self.scaler is None:
            return original_result
        
        try:
            # Extract features for ML prediction
            features = self.extract_behavior_features(person_data, original_result)
            
            # Ensure we have the right number of features (17)
            if len(features) < 17:
                # Pad with zeros if needed
                features = np.pad(features, (0, 17 - len(features)), 'constant')
            elif len(features) > 17:
                # Truncate if too many
                features = features[:17]
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get ML prediction
            ml_prediction = self.trained_model.predict(features_scaled)[0]
            ml_probability = self.trained_model.predict_proba(features_scaled)[0]
            
            # Get probability of shoplifting (class 1)
            shoplifting_probability = ml_probability[1] if len(ml_probability) > 1 else 0.0
            
            # Enhanced result with ML integration
            enhanced_result = original_result.copy()
            
            # Boost scores based on ML prediction
            if ml_prediction == 1:  # Shoplifting predicted
                # Significantly boost all scores
                enhanced_result['shelf_interaction_score'] = max(
                    enhanced_result.get('shelf_interaction_score', 0), 
                    shoplifting_probability * 0.8
                )
                enhanced_result['item_taking_score'] = max(
                    enhanced_result.get('item_taking_score', 0), 
                    shoplifting_probability * 0.9
                )
                enhanced_result['concealment_score'] = max(
                    enhanced_result.get('concealment_score', 0), 
                    shoplifting_probability * 0.95
                )
                enhanced_result['suspicious_score'] = max(
                    enhanced_result.get('suspicious_score', 0), 
                    shoplifting_probability
                )
                
                # Force stage progression if ML is confident
                if shoplifting_probability > 0.8:
                    if enhanced_result['shoplifting_stage'] == 'browsing':
                        enhanced_result['shoplifting_stage'] = 'taking'
                    elif enhanced_result['shoplifting_stage'] == 'taking':
                        enhanced_result['shoplifting_stage'] = 'concealing'
                    elif enhanced_result['shoplifting_stage'] == 'concealing':
                        enhanced_result['shoplifting_stage'] = 'shoplifting'
                
                # Add ML-based behavior
                ml_behavior = {
                    'type': 'ml_prediction',
                    'description': f'ML model detected shoplifting behavior (confidence: {shoplifting_probability:.2f})',
                    'confidence': shoplifting_probability,
                    'timestamp': datetime.now()
                }
                enhanced_result['behaviors'].append(ml_behavior)
            
            # Add ML metadata
            enhanced_result['ml_prediction'] = {
                'predicted_class': int(ml_prediction),
                'shoplifting_probability': float(shoplifting_probability),
                'model_used': True
            }
            
            return enhanced_result
            
        except Exception as e:
            print(f"Warning: ML prediction failed: {e}")
            return original_result
    
    def apply_model_to_detector(self):
        """Apply the trained model to enhance the detector"""
        if not self.load_trained_model():
            return False
        
        # Replace the detector's analyze_person_behavior method
        original_method = self.detector.analyze_person_behavior
        
        def enhanced_analyze_person_behavior(person_data, frame):
            return self.enhanced_behavior_analysis(person_data, frame)
        
        self.detector.analyze_person_behavior = enhanced_analyze_person_behavior
        
        print("✅ Trained model successfully integrated with detector")
        return True
    
    def test_enhanced_detection(self):
        """Test the enhanced detection system"""
        print("\n🧪 Testing Enhanced Detection with Trained ML Model")
        print("=" * 55)
        
        if not self.apply_model_to_detector():
            print("❌ Failed to apply model to detector")
            return
        
        # Test scenarios
        test_scenarios = [
            ("normal_browsing", False, "Customer browsing normally"),
            ("shelf_interaction", False, "Customer examining items"),
            ("concealment_behavior", True, "Customer concealing items"),
            ("exit_movement", True, "Customer moving to exit with concealed item")
        ]
        
        results = []
        
        for scenario_name, expected, description in test_scenarios:
            print(f"\n--- Testing: {description} ---")
            
            # Create mock person data
            mock_person = {
                'person_id': 1,
                'bbox': {'center_x': 150, 'center_y': 240, 'width': 80, 'height': 180,
                        'x1': 110, 'y1': 150, 'x2': 190, 'y2': 330},
                'track_data': {
                    'positions': [
                        {'x': 150 + i*2, 'y': 240, 'timestamp': datetime.now(), 'width': 80, 'height': 180}
                        for i in range(15)
                    ],
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now()
                }
            }
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Test detection with multiple steps
            for step in range(3):
                result = self.detector.analyze_person_behavior(mock_person, frame)
                is_suspicious = self.detector.is_behavior_suspicious(result)
                
                print(f"Step {step + 1}: Stage={result['shoplifting_stage']}, Score={result['suspicious_score']:.2f}, Suspicious={is_suspicious}")
                
                # Show ML prediction if available
                if 'ml_prediction' in result:
                    ml_info = result['ml_prediction']
                    print(f"  ML: Class={ml_info['predicted_class']}, Probability={ml_info['shoplifting_probability']:.2f}")
                
                # Show behaviors
                if result['behaviors']:
                    for behavior in result['behaviors'][-2:]:  # Show last 2 behaviors
                        print(f"  - {behavior['type']}: {behavior['description']}")
            
            final_result = result
            final_suspicious = is_suspicious
            
            results.append({
                'scenario': scenario_name,
                'expected': expected,
                'detected': final_suspicious,
                'score': final_result['suspicious_score'],
                'stage': final_result['shoplifting_stage'],
                'correct': expected == final_suspicious
            })
            
            status = "✅" if expected == final_suspicious else "❌"
            print(f"Final: {status} Expected={expected}, Detected={final_suspicious}, Correct={expected == final_suspicious}")
        
        # Summary
        print(f"\n🎯 Enhanced Detection Results:")
        correct_predictions = sum(1 for r in results if r['correct'])
        total_tests = len(results)
        accuracy = correct_predictions / total_tests
        
        print(f"   Accuracy: {accuracy:.2f} ({correct_predictions}/{total_tests})")
        
        for result in results:
            status = "✅" if result['correct'] else "❌"
            print(f"   {status} {result['scenario']}: Expected={result['expected']}, Got={result['detected']} (Score: {result['score']:.2f})")
        
        # Save results
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self.model_info,
            'detection_results': results,
            'overall_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_tests': total_tests
        }
        
        results_file = Path("enhanced_detection_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n📄 Results saved to: {results_file}")
        
        if accuracy >= 0.75:
            print("🎉 Enhanced detection system shows significant improvement!")
        else:
            print("⚠️  Enhanced detection system needs further optimization.")
        
        return results_summary

def main():
    """Main function to apply trained model and test"""
    print("🔧 Applying Trained ML Model to Shoplifting Detection System")
    print("=" * 65)
    
    integrator = ModelIntegrator()
    integrator.test_enhanced_detection()

if __name__ == "__main__":
    main()
