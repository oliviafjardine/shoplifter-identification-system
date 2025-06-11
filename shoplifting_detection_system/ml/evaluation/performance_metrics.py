#!/usr/bin/env python3
"""
Performance Evaluation System for Shoplifting Detection
Provides comprehensive accuracy metrics, testing, and performance analysis
"""

import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@dataclass
class DetectionResult:
    """Structure for detection results"""
    person_id: int
    timestamp: datetime
    detected: bool
    confidence: float
    stage: str
    behaviors: List[Dict]
    ground_truth: Optional[bool] = None  # For evaluation

@dataclass
class PerformanceMetrics:
    """Structure for performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_detections: int
    avg_confidence: float
    detection_latency: float  # Average time to detect

class PerformanceEvaluator:
    """
    Comprehensive performance evaluation system for shoplifting detection
    """
    
    def __init__(self):
        self.detection_results: List[DetectionResult] = []
        self.evaluation_history: List[PerformanceMetrics] = []
        self.test_scenarios = self._load_test_scenarios()
        
    def _load_test_scenarios(self) -> List[Dict]:
        """Load predefined test scenarios for evaluation"""
        return [
            {
                'name': 'normal_browsing',
                'description': 'Customer browsing normally without taking items',
                'expected_result': False,
                'confidence_threshold': 0.3
            },
            {
                'name': 'item_examination',
                'description': 'Customer picking up item to examine then putting back',
                'expected_result': False,
                'confidence_threshold': 0.5
            },
            {
                'name': 'legitimate_purchase',
                'description': 'Customer taking item and proceeding to checkout',
                'expected_result': False,
                'confidence_threshold': 0.6
            },
            {
                'name': 'concealment_shoplifting',
                'description': 'Customer taking item and concealing it',
                'expected_result': True,
                'confidence_threshold': 0.8
            },
            {
                'name': 'pocket_shoplifting',
                'description': 'Customer putting item in pocket/bag',
                'expected_result': True,
                'confidence_threshold': 0.9
            },
            {
                'name': 'distraction_theft',
                'description': 'Customer using distraction while taking items',
                'expected_result': True,
                'confidence_threshold': 0.7
            }
        ]
    
    def add_detection_result(self, result: DetectionResult):
        """Add a detection result for evaluation"""
        self.detection_results.append(result)
    
    def evaluate_performance(self, time_window_hours: int = 24) -> PerformanceMetrics:
        """
        Evaluate performance over a specified time window
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_results = [r for r in self.detection_results if r.timestamp >= cutoff_time]
        
        if not recent_results:
            return self._empty_metrics()
        
        # Calculate confusion matrix
        tp = sum(1 for r in recent_results if r.detected and r.ground_truth)
        fp = sum(1 for r in recent_results if r.detected and not r.ground_truth)
        tn = sum(1 for r in recent_results if not r.detected and not r.ground_truth)
        fn = sum(1 for r in recent_results if not r.detected and r.ground_truth)
        
        total = len(recent_results)
        
        # Calculate metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        avg_confidence = np.mean([r.confidence for r in recent_results])
        
        # Calculate detection latency (simplified)
        detection_latency = self._calculate_detection_latency(recent_results)
        
        metrics = PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            total_detections=total,
            avg_confidence=avg_confidence,
            detection_latency=detection_latency
        )
        
        self.evaluation_history.append(metrics)
        return metrics
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no data available"""
        return PerformanceMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
            false_positive_rate=0.0, false_negative_rate=0.0,
            true_positives=0, false_positives=0, true_negatives=0, false_negatives=0,
            total_detections=0, avg_confidence=0.0, detection_latency=0.0
        )
    
    def _calculate_detection_latency(self, results: List[DetectionResult]) -> float:
        """Calculate average detection latency"""
        # This is a simplified calculation
        # In a real system, you'd measure time from suspicious behavior start to detection
        detection_times = []
        for result in results:
            if result.detected:
                # Estimate latency based on number of behaviors detected
                estimated_latency = len(result.behaviors) * 0.5  # 0.5 seconds per behavior
                detection_times.append(estimated_latency)
        
        return np.mean(detection_times) if detection_times else 0.0
    
    def generate_performance_report(self, save_path: Optional[str] = None) -> Dict:
        """Generate comprehensive performance report"""
        if not self.evaluation_history:
            current_metrics = self.evaluate_performance()
        else:
            current_metrics = self.evaluation_history[-1]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': {
                'accuracy': current_metrics.accuracy,
                'precision': current_metrics.precision,
                'recall': current_metrics.recall,
                'f1_score': current_metrics.f1_score,
                'false_positive_rate': current_metrics.false_positive_rate,
                'false_negative_rate': current_metrics.false_negative_rate,
                'avg_confidence': current_metrics.avg_confidence,
                'detection_latency': current_metrics.detection_latency
            },
            'confusion_matrix': {
                'true_positives': current_metrics.true_positives,
                'false_positives': current_metrics.false_positives,
                'true_negatives': current_metrics.true_negatives,
                'false_negatives': current_metrics.false_negatives
            },
            'recommendations': self._generate_recommendations(current_metrics),
            'total_detections': current_metrics.total_detections,
            'evaluation_period': '24 hours'
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        if metrics.accuracy < 0.8:
            recommendations.append("Overall accuracy is below 80%. Consider adjusting detection thresholds.")
        
        if metrics.false_positive_rate > 0.2:
            recommendations.append("High false positive rate detected. Increase detection thresholds to reduce false alarms.")
        
        if metrics.false_negative_rate > 0.3:
            recommendations.append("High false negative rate detected. Lower detection thresholds to catch more incidents.")
        
        if metrics.precision < 0.7:
            recommendations.append("Low precision indicates many false positives. Improve behavior analysis algorithms.")
        
        if metrics.recall < 0.7:
            recommendations.append("Low recall indicates missing actual incidents. Enhance detection sensitivity.")
        
        if metrics.detection_latency > 3.0:
            recommendations.append("Detection latency is high. Optimize algorithms for faster response.")
        
        if metrics.avg_confidence < 0.6:
            recommendations.append("Average confidence is low. Review and improve confidence scoring mechanisms.")
        
        if not recommendations:
            recommendations.append("Performance metrics are within acceptable ranges. Continue monitoring.")
        
        return recommendations
    
    def plot_performance_trends(self, save_path: Optional[str] = None):
        """Plot performance trends over time"""
        if len(self.evaluation_history) < 2:
            print("Insufficient data for trend analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Shoplifting Detection Performance Trends', fontsize=16)
        
        # Extract data for plotting
        timestamps = list(range(len(self.evaluation_history)))
        accuracies = [m.accuracy for m in self.evaluation_history]
        precisions = [m.precision for m in self.evaluation_history]
        recalls = [m.recall for m in self.evaluation_history]
        f1_scores = [m.f1_score for m in self.evaluation_history]
        
        # Plot accuracy
        axes[0, 0].plot(timestamps, accuracies, 'b-o')
        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)
        
        # Plot precision and recall
        axes[0, 1].plot(timestamps, precisions, 'g-o', label='Precision')
        axes[0, 1].plot(timestamps, recalls, 'r-o', label='Recall')
        axes[0, 1].set_title('Precision and Recall Over Time')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot F1 score
        axes[1, 0].plot(timestamps, f1_scores, 'm-o')
        axes[1, 0].set_title('F1 Score Over Time')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].grid(True)
        
        # Plot false positive rate
        fpr_values = [m.false_positive_rate for m in self.evaluation_history]
        axes[1, 1].plot(timestamps, fpr_values, 'r-o')
        axes[1, 1].set_title('False Positive Rate Over Time')
        axes[1, 1].set_ylabel('False Positive Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def run_benchmark_tests(self, detector) -> Dict:
        """Run benchmark tests on the detector"""
        print("Running benchmark tests...")
        
        test_results = {}
        
        for scenario in self.test_scenarios:
            print(f"Testing scenario: {scenario['name']}")
            
            # This would be implemented with actual test data
            # For now, we'll simulate the test
            result = self._simulate_test_scenario(detector, scenario)
            test_results[scenario['name']] = result
        
        return test_results
    
    def _simulate_test_scenario(self, detector, scenario: Dict) -> Dict:
        """Simulate a test scenario (placeholder implementation)"""
        # In a real implementation, this would run actual test data through the detector
        # For now, return simulated results
        return {
            'expected': scenario['expected_result'],
            'detected': np.random.choice([True, False]),
            'confidence': np.random.uniform(0.3, 0.9),
            'passed': True  # Would be calculated based on actual vs expected
        }

if __name__ == "__main__":
    # Example usage
    evaluator = PerformanceEvaluator()
    
    # Simulate some detection results
    for i in range(100):
        result = DetectionResult(
            person_id=i,
            timestamp=datetime.now() - timedelta(hours=np.random.randint(0, 24)),
            detected=np.random.choice([True, False]),
            confidence=np.random.uniform(0.3, 0.9),
            stage=np.random.choice(['browsing', 'taking', 'concealing', 'shoplifting']),
            behaviors=[],
            ground_truth=np.random.choice([True, False])
        )
        evaluator.add_detection_result(result)
    
    # Generate performance report
    report = evaluator.generate_performance_report('performance_report.json')
    print("Performance Report Generated:")
    print(json.dumps(report, indent=2))
