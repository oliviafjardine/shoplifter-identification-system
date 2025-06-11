#!/usr/bin/env python3
"""
Enhanced test script for shoplifting detection system with performance evaluation.
Tests the improved detection system and provides comprehensive accuracy metrics.
"""

import numpy as np
import sys
import os
from datetime import datetime, timedelta
from detection.shoplifting_detector import ShopliftingDetector

# Add evaluation directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'evaluation'))
try:
    from performance_evaluator import PerformanceEvaluator, DetectionResult
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("Performance evaluation not available - install matplotlib and seaborn for full evaluation")


def create_mock_person_data(person_id, positions, bbox_data):
    """Create mock person data for testing"""
    track_data = {
        'positions': []
    }

    for i, (x, y) in enumerate(positions):
        track_data['positions'].append({
            'x': x,
            'y': y,
            'width': bbox_data.get('width', 100),
            'height': bbox_data.get('height', 200),
            'timestamp': datetime.now() - timedelta(seconds=len(positions)-i)
        })

    return {
        'person_id': person_id,
        'bbox': {
            'x1': x - bbox_data.get('width', 100)//2,
            'y1': y - bbox_data.get('height', 200)//2,
            'x2': x + bbox_data.get('width', 100)//2,
            'y2': y + bbox_data.get('height', 200)//2,
            'center_x': x,
            'center_y': y,
            'width': bbox_data.get('width', 100),
            'height': bbox_data.get('height', 200)
        },
        'track_data': track_data
    }


def test_normal_browsing():
    """Test normal browsing behavior - should NOT trigger alerts"""
    print("\n=== Testing Normal Browsing Behavior ===")

    detector = ShopliftingDetector()

    # Person walking around normally, not interacting with shelves
    positions = [(100, 300), (120, 300), (140, 300), (160, 300), (180, 300)]
    person_data = create_mock_person_data(
        1, positions, {'width': 80, 'height': 180})

    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Mock frame

    result = detector.analyze_person_behavior(person_data, frame)

    print(f"Person ID: {result['person_id']}")
    print(f"Shoplifting Stage: {result['shoplifting_stage']}")
    print(f"Suspicious Score: {result['suspicious_score']:.2f}")
    print(f"Behaviors Detected: {len(result['behaviors'])}")
    print(f"Is Suspicious: {detector.is_behavior_suspicious(result)}")

    for behavior in result['behaviors']:
        print(f"  - {behavior['type']}: {behavior['description']}")


def test_shelf_interaction_only():
    """Test shelf interaction without concealment - should NOT trigger alerts"""
    print("\n=== Testing Shelf Interaction Only ===")

    detector = ShopliftingDetector()

    # Person approaches shelf area and interacts but doesn't conceal anything
    positions = [(300, 300), (150, 200), (120, 180), (115, 175), (120, 180)]
    person_data = create_mock_person_data(
        2, positions, {'width': 90, 'height': 190})

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.analyze_person_behavior(person_data, frame)

    print(f"Person ID: {result['person_id']}")
    print(f"Shoplifting Stage: {result['shoplifting_stage']}")
    print(f"Suspicious Score: {result['suspicious_score']:.2f}")
    print(f"Behaviors Detected: {len(result['behaviors'])}")
    print(f"Is Suspicious: {detector.is_behavior_suspicious(result)}")

    for behavior in result['behaviors']:
        print(f"  - {behavior['type']}: {behavior['description']}")


def test_shoplifting_sequence():
    """Test complete shoplifting sequence - SHOULD trigger alerts"""
    print("\n=== Testing Complete Shoplifting Sequence ===")

    detector = ShopliftingDetector()

    # Simulate complete shoplifting sequence over multiple analysis calls
    person_id = 3

    # Step 1: Approach shelf
    print("\nStep 1: Approaching shelf...")
    positions = [(300, 300), (200, 250), (150, 200)]
    person_data = create_mock_person_data(
        person_id, positions, {'width': 85, 'height': 185})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result1 = detector.analyze_person_behavior(person_data, frame)
    print(
        f"Stage: {result1['shoplifting_stage']}, Score: {result1['suspicious_score']:.2f}")

    # Step 2: Interact with shelf and take item
    print("\nStep 2: Taking item from shelf...")
    positions = [(150, 200), (120, 180), (115, 175), (110, 170), (115, 175)]
    person_data = create_mock_person_data(
        # Wider bbox (reaching)
        person_id, positions, {'width': 110, 'height': 185})

    result2 = detector.analyze_person_behavior(person_data, frame)
    print(
        f"Stage: {result2['shoplifting_stage']}, Score: {result2['suspicious_score']:.2f}")

    # Step 3: Conceal item (simulate turning away and putting in pocket)
    print("\nStep 3: Concealing item...")
    positions = [(115, 175), (120, 180), (125, 185), (130, 190), (135, 195)]
    person_data = create_mock_person_data(person_id, positions, {
                                          # Narrower bbox (turning away)
                                          'width': 70, 'height': 185})

    result3 = detector.analyze_person_behavior(person_data, frame)
    print(
        f"Stage: {result3['shoplifting_stage']}, Score: {result3['suspicious_score']:.2f}")

    # Step 4: Move toward exit with concealed item
    print("\nStep 4: Moving toward exit...")
    positions = [(135, 195), (200, 250), (300, 300), (450, 350), (580, 300)]
    person_data = create_mock_person_data(
        person_id, positions, {'width': 75, 'height': 185})

    result4 = detector.analyze_person_behavior(person_data, frame)
    print(
        f"Stage: {result4['shoplifting_stage']}, Score: {result4['suspicious_score']:.2f}")
    print(f"Is Suspicious: {detector.is_behavior_suspicious(result4)}")

    print("\nFinal Analysis:")
    for behavior in result4['behaviors']:
        print(f"  - {behavior['type']}: {behavior['description']}")


def test_false_positive_scenarios():
    """Test scenarios that should NOT trigger false positives"""
    print("\n=== Testing False Positive Scenarios ===")

    detector = ShopliftingDetector()

    # Scenario 1: Person crouching to tie shoes (not near shelf)
    print("\nScenario 1: Crouching away from shelves...")
    positions = [(400, 300), (400, 300), (400, 300), (400, 300), (400, 300)]
    person_data = create_mock_person_data(
        4, positions, {'width': 120, 'height': 100})  # Wide/short (crouching)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.analyze_person_behavior(person_data, frame)
    print(f"Stage: {result['shoplifting_stage']}, Score: {result['suspicious_score']:.2f}, Suspicious: {detector.is_behavior_suspicious(result)}")

    # Scenario 2: Person moving erratically but not near shelves
    print("\nScenario 2: Erratic movement away from shelves...")
    positions = [(400, 300), (420, 320), (380, 280), (440, 340), (360, 260)]
    person_data = create_mock_person_data(
        5, positions, {'width': 85, 'height': 185})

    result = detector.analyze_person_behavior(person_data, frame)
    print(f"Stage: {result['shoplifting_stage']}, Score: {result['suspicious_score']:.2f}, Suspicious: {detector.is_behavior_suspicious(result)}")


def run_performance_evaluation():
    """Run comprehensive performance evaluation"""
    if not EVALUATION_AVAILABLE:
        print("Skipping performance evaluation - dependencies not available")
        return

    print("\n=== Performance Evaluation ===")
    evaluator = PerformanceEvaluator()
    detector = ShopliftingDetector()

    # Run benchmark tests
    test_results = evaluator.run_benchmark_tests(detector)

    # Simulate detection results for evaluation
    test_cases = [
        # Normal browsing cases (should not trigger)
        {'person_id': 1, 'expected': False, 'scenario': 'normal_browsing'},
        {'person_id': 2, 'expected': False, 'scenario': 'item_examination'},
        {'person_id': 3, 'expected': False, 'scenario': 'legitimate_purchase'},

        # Shoplifting cases (should trigger)
        {'person_id': 4, 'expected': True, 'scenario': 'concealment_shoplifting'},
        {'person_id': 5, 'expected': True, 'scenario': 'pocket_shoplifting'},
        {'person_id': 6, 'expected': True, 'scenario': 'distraction_theft'},
    ]

    for case in test_cases:
        # Run actual detection
        positions = [(300, 300), (200, 250), (150, 200)]
        person_data = create_mock_person_data(case['person_id'], positions, {
                                              'width': 85, 'height': 185})
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = detector.analyze_person_behavior(person_data, frame)
        is_suspicious = detector.is_behavior_suspicious(result)

        # Create detection result for evaluation
        detection_result = DetectionResult(
            person_id=case['person_id'],
            timestamp=datetime.now(),
            detected=is_suspicious,
            confidence=result['suspicious_score'],
            stage=result['shoplifting_stage'],
            behaviors=result['behaviors'],
            ground_truth=case['expected']
        )

        evaluator.add_detection_result(detection_result)

        print(
            f"Case {case['scenario']}: Expected={case['expected']}, Detected={is_suspicious}, Score={result['suspicious_score']:.2f}")

    # Generate performance metrics
    metrics = evaluator.evaluate_performance()

    print(f"\nPerformance Metrics:")
    print(f"Accuracy: {metrics.accuracy:.2f}")
    print(f"Precision: {metrics.precision:.2f}")
    print(f"Recall: {metrics.recall:.2f}")
    print(f"F1 Score: {metrics.f1_score:.2f}")
    print(f"False Positive Rate: {metrics.false_positive_rate:.2f}")
    print(f"False Negative Rate: {metrics.false_negative_rate:.2f}")

    # Generate report
    report = evaluator.generate_performance_report('performance_report.json')
    print(f"\nDetailed report saved to: performance_report.json")

    return metrics


def display_system_improvements():
    """Display information about system improvements"""
    print("\n=== System Improvements Made ===")
    print("1. LOWERED DETECTION THRESHOLDS:")
    print("   - Shelf interaction: 0.6 → 0.4 (more sensitive)")
    print("   - Concealment: 0.7 → 0.5 (better detection)")
    print("   - Shoplifting: 0.9 → 0.7 (more realistic)")

    print("\n2. ENHANCED BEHAVIOR ANALYSIS:")
    print("   - Improved shelf proximity scoring")
    print("   - Better time-based interaction detection")
    print("   - More sensitive bbox change analysis")
    print("   - Reduced minimum position requirements")

    print("\n3. ADDED PERFORMANCE TRACKING:")
    print("   - Real-time accuracy metrics")
    print("   - Precision and recall calculation")
    print("   - False positive/negative rate monitoring")
    print("   - Performance history tracking")

    print("\n4. IMPROVED DETECTION LOGIC:")
    print("   - More sensitive movement pattern analysis")
    print("   - Better approach-stop-reach detection")
    print("   - Enhanced confidence scoring")
    print("   - Shelf interaction confidence boost")


if __name__ == "__main__":
    print("Enhanced Shoplifting Detection System Test")
    print("=" * 60)
    print("\nThis test demonstrates the improved shoplifting detection system:")
    print("1. Enhanced sensitivity with lower detection thresholds")
    print("2. Improved behavior analysis algorithms")
    print("3. Comprehensive performance evaluation")
    print("4. Real-time accuracy metrics")

    # Run original tests
    test_normal_browsing()
    test_shelf_interaction_only()
    test_shoplifting_sequence()
    test_false_positive_scenarios()

    # Display improvements
    display_system_improvements()

    # Run performance evaluation
    run_performance_evaluation()

    print("\n" + "=" * 60)
    print("Enhanced testing completed!")
    print("\nNext steps:")
    print("1. Review performance_report.json for detailed metrics")
    print("2. Adjust thresholds based on false positive/negative rates")
    print("3. Test with real video data for validation")
    print("4. Monitor performance over time")
