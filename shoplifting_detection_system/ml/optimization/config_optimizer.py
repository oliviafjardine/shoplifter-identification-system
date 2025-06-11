#!/usr/bin/env python3
"""
Configuration Optimizer for Shoplifting Detection System
Automatically adjusts detection thresholds based on performance metrics
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from detection.shoplifting_detector import ShopliftingDetector

class ConfigurationOptimizer:
    """
    Automatically optimize detection thresholds based on performance feedback
    """
    
    def __init__(self):
        self.optimization_history = []
        self.current_config = self._get_default_config()
        self.performance_targets = {
            'min_accuracy': 0.85,
            'max_false_positive_rate': 0.15,
            'max_false_negative_rate': 0.20,
            'min_precision': 0.80,
            'min_recall': 0.75
        }
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'shelf_interaction_threshold': 0.4,
            'concealment_threshold': 0.5,
            'shoplifting_threshold': 0.7,
            'alert_threshold': 0.7,
            'confidence_boost_factor': 0.3,
            'proximity_weight': 0.3,
            'time_weight': 0.3,
            'bbox_change_threshold': 0.2
        }
    
    def optimize_thresholds(self, performance_metrics: Dict) -> Dict:
        """
        Optimize detection thresholds based on performance metrics
        """
        print("Optimizing detection thresholds...")
        
        current_accuracy = performance_metrics.get('accuracy', 0.0)
        current_fpr = performance_metrics.get('false_positive_rate', 0.0)
        current_fnr = performance_metrics.get('false_negative_rate', 0.0)
        current_precision = performance_metrics.get('precision', 0.0)
        current_recall = performance_metrics.get('recall', 0.0)
        
        optimization_suggestions = []
        new_config = self.current_config.copy()
        
        # Optimize based on false positive rate
        if current_fpr > self.performance_targets['max_false_positive_rate']:
            print(f"High false positive rate ({current_fpr:.2f}). Increasing thresholds...")
            
            # Increase thresholds to reduce false positives
            new_config['shelf_interaction_threshold'] = min(0.8, new_config['shelf_interaction_threshold'] + 0.1)
            new_config['concealment_threshold'] = min(0.8, new_config['concealment_threshold'] + 0.1)
            new_config['shoplifting_threshold'] = min(0.9, new_config['shoplifting_threshold'] + 0.1)
            new_config['bbox_change_threshold'] = min(0.5, new_config['bbox_change_threshold'] + 0.05)
            
            optimization_suggestions.append("Increased detection thresholds to reduce false positives")
        
        # Optimize based on false negative rate
        elif current_fnr > self.performance_targets['max_false_negative_rate']:
            print(f"High false negative rate ({current_fnr:.2f}). Decreasing thresholds...")
            
            # Decrease thresholds to catch more incidents
            new_config['shelf_interaction_threshold'] = max(0.2, new_config['shelf_interaction_threshold'] - 0.1)
            new_config['concealment_threshold'] = max(0.3, new_config['concealment_threshold'] - 0.1)
            new_config['shoplifting_threshold'] = max(0.5, new_config['shoplifting_threshold'] - 0.1)
            new_config['bbox_change_threshold'] = max(0.1, new_config['bbox_change_threshold'] - 0.05)
            
            optimization_suggestions.append("Decreased detection thresholds to catch more incidents")
        
        # Optimize based on precision
        if current_precision < self.performance_targets['min_precision']:
            print(f"Low precision ({current_precision:.2f}). Adjusting confidence scoring...")
            
            # Increase confidence requirements
            new_config['confidence_boost_factor'] = min(0.5, new_config['confidence_boost_factor'] + 0.1)
            new_config['proximity_weight'] = min(0.5, new_config['proximity_weight'] + 0.1)
            
            optimization_suggestions.append("Increased confidence requirements to improve precision")
        
        # Optimize based on recall
        if current_recall < self.performance_targets['min_recall']:
            print(f"Low recall ({current_recall:.2f}). Enhancing sensitivity...")
            
            # Increase sensitivity
            new_config['time_weight'] = min(0.5, new_config['time_weight'] + 0.1)
            new_config['proximity_weight'] = min(0.5, new_config['proximity_weight'] + 0.1)
            
            optimization_suggestions.append("Enhanced sensitivity to improve recall")
        
        # Record optimization
        optimization_record = {
            'timestamp': datetime.now().isoformat(),
            'previous_config': self.current_config,
            'new_config': new_config,
            'performance_metrics': performance_metrics,
            'suggestions': optimization_suggestions
        }
        
        self.optimization_history.append(optimization_record)
        self.current_config = new_config
        
        return {
            'optimized_config': new_config,
            'suggestions': optimization_suggestions,
            'improvements_expected': self._predict_improvements(performance_metrics, new_config)
        }
    
    def _predict_improvements(self, current_metrics: Dict, new_config: Dict) -> Dict:
        """Predict expected improvements from configuration changes"""
        predictions = {}
        
        # Simple heuristic predictions
        if new_config['shelf_interaction_threshold'] > self.current_config['shelf_interaction_threshold']:
            predictions['false_positive_rate'] = 'Expected to decrease'
            predictions['recall'] = 'May slightly decrease'
        
        if new_config['concealment_threshold'] < self.current_config['concealment_threshold']:
            predictions['recall'] = 'Expected to increase'
            predictions['false_positive_rate'] = 'May slightly increase'
        
        if new_config['confidence_boost_factor'] > self.current_config['confidence_boost_factor']:
            predictions['precision'] = 'Expected to increase'
        
        return predictions
    
    def apply_configuration(self, detector: ShopliftingDetector) -> bool:
        """Apply optimized configuration to detector"""
        try:
            detector.shelf_interaction_threshold = self.current_config['shelf_interaction_threshold']
            detector.concealment_threshold = self.current_config['concealment_threshold']
            detector.shoplifting_threshold = self.current_config['shoplifting_threshold']
            
            print("Configuration applied successfully!")
            return True
        except Exception as e:
            print(f"Error applying configuration: {e}")
            return False
    
    def save_optimization_history(self, filepath: str = 'optimization_history.json'):
        """Save optimization history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.optimization_history, f, indent=2)
        print(f"Optimization history saved to {filepath}")
    
    def generate_optimization_report(self) -> Dict:
        """Generate comprehensive optimization report"""
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
        
        latest_optimization = self.optimization_history[-1]
        
        report = {
            'current_configuration': self.current_config,
            'optimization_count': len(self.optimization_history),
            'latest_optimization': latest_optimization,
            'performance_targets': self.performance_targets,
            'configuration_trends': self._analyze_configuration_trends()
        }
        
        return report
    
    def _analyze_configuration_trends(self) -> Dict:
        """Analyze trends in configuration changes"""
        if len(self.optimization_history) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        trends = {}
        
        # Analyze threshold trends
        shelf_thresholds = [opt['new_config']['shelf_interaction_threshold'] for opt in self.optimization_history]
        concealment_thresholds = [opt['new_config']['concealment_threshold'] for opt in self.optimization_history]
        shoplifting_thresholds = [opt['new_config']['shoplifting_threshold'] for opt in self.optimization_history]
        
        trends['shelf_interaction_threshold'] = {
            'trend': 'increasing' if shelf_thresholds[-1] > shelf_thresholds[0] else 'decreasing',
            'change': shelf_thresholds[-1] - shelf_thresholds[0]
        }
        
        trends['concealment_threshold'] = {
            'trend': 'increasing' if concealment_thresholds[-1] > concealment_thresholds[0] else 'decreasing',
            'change': concealment_thresholds[-1] - concealment_thresholds[0]
        }
        
        trends['shoplifting_threshold'] = {
            'trend': 'increasing' if shoplifting_thresholds[-1] > shoplifting_thresholds[0] else 'decreasing',
            'change': shoplifting_thresholds[-1] - shoplifting_thresholds[0]
        }
        
        return trends
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.current_config = self._get_default_config()
        print("Configuration reset to defaults")
    
    def get_recommended_config_for_environment(self, environment_type: str) -> Dict:
        """Get recommended configuration for specific environment"""
        configs = {
            'high_traffic_store': {
                'shelf_interaction_threshold': 0.5,
                'concealment_threshold': 0.6,
                'shoplifting_threshold': 0.8,
                'description': 'Higher thresholds to reduce false positives in busy environments'
            },
            'small_store': {
                'shelf_interaction_threshold': 0.3,
                'concealment_threshold': 0.4,
                'shoplifting_threshold': 0.6,
                'description': 'Lower thresholds for better detection in smaller spaces'
            },
            'jewelry_store': {
                'shelf_interaction_threshold': 0.2,
                'concealment_threshold': 0.3,
                'shoplifting_threshold': 0.5,
                'description': 'Very sensitive settings for high-value items'
            },
            'grocery_store': {
                'shelf_interaction_threshold': 0.4,
                'concealment_threshold': 0.5,
                'shoplifting_threshold': 0.7,
                'description': 'Balanced settings for grocery environments'
            }
        }
        
        return configs.get(environment_type, self._get_default_config())

def main():
    """Example usage of configuration optimizer"""
    optimizer = ConfigurationOptimizer()
    
    # Simulate performance metrics
    sample_metrics = {
        'accuracy': 0.75,
        'precision': 0.70,
        'recall': 0.80,
        'false_positive_rate': 0.25,
        'false_negative_rate': 0.15
    }
    
    print("Current Configuration:")
    print(json.dumps(optimizer.current_config, indent=2))
    
    # Optimize configuration
    optimization_result = optimizer.optimize_thresholds(sample_metrics)
    
    print("\nOptimization Results:")
    print(json.dumps(optimization_result, indent=2))
    
    # Generate report
    report = optimizer.generate_optimization_report()
    print("\nOptimization Report:")
    print(json.dumps(report, indent=2))
    
    # Save history
    optimizer.save_optimization_history()

if __name__ == "__main__":
    main()
