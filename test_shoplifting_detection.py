#!/usr/bin/env python3
"""
Test script to demonstrate the new shoplifting detection system.
This shows how the system now focuses on actual shoplifting behavior
rather than general suspicious activities.
"""

import numpy as np
from datetime import datetime, timedelta
from detection.shoplifting_detector import ShopliftingDetector


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
    person_data = create_mock_person_data(1, positions, {'width': 80, 'height': 180})
    
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
    person_data = create_mock_person_data(2, positions, {'width': 90, 'height': 190})
    
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
    person_data = create_mock_person_data(person_id, positions, {'width': 85, 'height': 185})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result1 = detector.analyze_person_behavior(person_data, frame)
    print(f"Stage: {result1['shoplifting_stage']}, Score: {result1['suspicious_score']:.2f}")
    
    # Step 2: Interact with shelf and take item
    print("\nStep 2: Taking item from shelf...")
    positions = [(150, 200), (120, 180), (115, 175), (110, 170), (115, 175)]
    person_data = create_mock_person_data(person_id, positions, {'width': 110, 'height': 185})  # Wider bbox (reaching)
    
    result2 = detector.analyze_person_behavior(person_data, frame)
    print(f"Stage: {result2['shoplifting_stage']}, Score: {result2['suspicious_score']:.2f}")
    
    # Step 3: Conceal item (simulate turning away and putting in pocket)
    print("\nStep 3: Concealing item...")
    positions = [(115, 175), (120, 180), (125, 185), (130, 190), (135, 195)]
    person_data = create_mock_person_data(person_id, positions, {'width': 70, 'height': 185})  # Narrower bbox (turning away)
    
    result3 = detector.analyze_person_behavior(person_data, frame)
    print(f"Stage: {result3['shoplifting_stage']}, Score: {result3['suspicious_score']:.2f}")
    
    # Step 4: Move toward exit with concealed item
    print("\nStep 4: Moving toward exit...")
    positions = [(135, 195), (200, 250), (300, 300), (450, 350), (580, 300)]
    person_data = create_mock_person_data(person_id, positions, {'width': 75, 'height': 185})
    
    result4 = detector.analyze_person_behavior(person_data, frame)
    print(f"Stage: {result4['shoplifting_stage']}, Score: {result4['suspicious_score']:.2f}")
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
    person_data = create_mock_person_data(4, positions, {'width': 120, 'height': 100})  # Wide/short (crouching)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = detector.analyze_person_behavior(person_data, frame)
    print(f"Stage: {result['shoplifting_stage']}, Score: {result['suspicious_score']:.2f}, Suspicious: {detector.is_behavior_suspicious(result)}")
    
    # Scenario 2: Person moving erratically but not near shelves
    print("\nScenario 2: Erratic movement away from shelves...")
    positions = [(400, 300), (420, 320), (380, 280), (440, 340), (360, 260)]
    person_data = create_mock_person_data(5, positions, {'width': 85, 'height': 185})
    
    result = detector.analyze_person_behavior(person_data, frame)
    print(f"Stage: {result['shoplifting_stage']}, Score: {result['suspicious_score']:.2f}, Suspicious: {detector.is_behavior_suspicious(result)}")


if __name__ == "__main__":
    print("Shoplifting Detection System Test")
    print("=" * 50)
    print("\nThis test demonstrates the new shoplifting detection logic:")
    print("1. Only alerts when actual shoplifting behavior is detected")
    print("2. Requires sequence: Take from shelf → Conceal → Intent to leave")
    print("3. Ignores general suspicious behavior not related to theft")
    
    test_normal_browsing()
    test_shelf_interaction_only()
    test_shoplifting_sequence()
    test_false_positive_scenarios()
    
    print("\n" + "=" * 50)
    print("Test completed. The system now focuses on actual shoplifting behavior!")
