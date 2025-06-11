#!/usr/bin/env python3
"""
Realistic test scenarios for shoplifting detection with proper mock data
"""

import numpy as np
from datetime import datetime, timedelta
from detection.shoplifting_detector import ShopliftingDetector

def create_realistic_person_data(person_id, scenario_type):
    """Create realistic person data for different scenarios"""
    
    if scenario_type == "normal_browsing":
        # Person walking around, not near shelves
        positions = [
            {'x': 400, 'y': 300, 'timestamp': datetime.now() - timedelta(seconds=10), 'width': 80, 'height': 180},
            {'x': 420, 'y': 320, 'timestamp': datetime.now() - timedelta(seconds=9), 'width': 82, 'height': 180},
            {'x': 440, 'y': 340, 'timestamp': datetime.now() - timedelta(seconds=8), 'width': 81, 'height': 180},
            {'x': 460, 'y': 360, 'timestamp': datetime.now() - timedelta(seconds=7), 'width': 80, 'height': 180},
            {'x': 480, 'y': 380, 'timestamp': datetime.now() - timedelta(seconds=6), 'width': 79, 'height': 180},
        ]
        
    elif scenario_type == "shelf_interaction":
        # Person approaching shelf, stopping, reaching
        positions = [
            {'x': 300, 'y': 300, 'timestamp': datetime.now() - timedelta(seconds=10), 'width': 80, 'height': 180},
            {'x': 250, 'y': 280, 'timestamp': datetime.now() - timedelta(seconds=9), 'width': 82, 'height': 180},
            {'x': 200, 'y': 260, 'timestamp': datetime.now() - timedelta(seconds=8), 'width': 85, 'height': 180},
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=7), 'width': 90, 'height': 180},  # Near shelf, reaching
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=6), 'width': 95, 'height': 180},  # Stationary, reaching
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=5), 'width': 90, 'height': 180},  # Still reaching
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=4), 'width': 85, 'height': 180},  # Retracting
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=3), 'width': 80, 'height': 180},  # Normal
        ]
        
    elif scenario_type == "concealment_behavior":
        # Person taking item and concealing it
        positions = [
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=15), 'width': 80, 'height': 180},
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=14), 'width': 90, 'height': 180},  # Reaching
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=13), 'width': 95, 'height': 180},  # Taking
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=12), 'width': 85, 'height': 180},  # Retracting
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=11), 'width': 70, 'height': 180},  # Turning away
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=10), 'width': 65, 'height': 180},  # More turning
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=9), 'width': 120, 'height': 100},   # Crouching/concealing
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=8), 'width': 110, 'height': 110},   # Still concealing
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=7), 'width': 80, 'height': 180},    # Standing up
            {'x': 170, 'y': 250, 'timestamp': datetime.now() - timedelta(seconds=6), 'width': 80, 'height': 180},    # Moving away
        ]
        
    elif scenario_type == "exit_movement":
        # Person with concealed item moving toward exit
        positions = [
            {'x': 150, 'y': 240, 'timestamp': datetime.now() - timedelta(seconds=20), 'width': 80, 'height': 180},
            {'x': 200, 'y': 260, 'timestamp': datetime.now() - timedelta(seconds=18), 'width': 80, 'height': 180},
            {'x': 250, 'y': 280, 'timestamp': datetime.now() - timedelta(seconds=16), 'width': 80, 'height': 180},
            {'x': 300, 'y': 300, 'timestamp': datetime.now() - timedelta(seconds=14), 'width': 80, 'height': 180},
            {'x': 350, 'y': 320, 'timestamp': datetime.now() - timedelta(seconds=12), 'width': 80, 'height': 180},
            {'x': 400, 'y': 340, 'timestamp': datetime.now() - timedelta(seconds=10), 'width': 80, 'height': 180},
            {'x': 450, 'y': 360, 'timestamp': datetime.now() - timedelta(seconds=8), 'width': 80, 'height': 180},
            {'x': 500, 'y': 380, 'timestamp': datetime.now() - timedelta(seconds=6), 'width': 80, 'height': 180},
            {'x': 550, 'y': 400, 'timestamp': datetime.now() - timedelta(seconds=4), 'width': 80, 'height': 180},
            {'x': 600, 'y': 420, 'timestamp': datetime.now() - timedelta(seconds=2), 'width': 80, 'height': 180},  # Near exit
        ]
    
    else:
        # Default case
        positions = [
            {'x': 300, 'y': 300, 'timestamp': datetime.now(), 'width': 80, 'height': 180}
        ]
    
    # Create person data structure
    latest_position = positions[-1]
    person_data = {
        'person_id': person_id,
        'bbox': {
            'x1': latest_position['x'] - latest_position['width']//2,
            'y1': latest_position['y'] - latest_position['height']//2,
            'x2': latest_position['x'] + latest_position['width']//2,
            'y2': latest_position['y'] + latest_position['height']//2,
            'width': latest_position['width'],
            'height': latest_position['height'],
            'center_x': latest_position['x'],
            'center_y': latest_position['y']
        },
        'track_data': {
            'positions': positions,
            'first_seen': positions[0]['timestamp'],
            'last_seen': positions[-1]['timestamp']
        }
    }
    
    return person_data

def test_realistic_scenarios():
    """Test realistic shoplifting scenarios"""
    print("=== Testing Realistic Scenarios ===")
    
    detector = ShopliftingDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    scenarios = [
        ("normal_browsing", False, "Customer browsing normally"),
        ("shelf_interaction", False, "Customer examining items on shelf"),
        ("concealment_behavior", True, "Customer concealing item after taking"),
        ("exit_movement", True, "Customer moving toward exit with concealed item")
    ]
    
    results = []
    
    for scenario_name, expected_suspicious, description in scenarios:
        print(f"\n--- {description} ---")
        
        person_data = create_realistic_person_data(len(results) + 1, scenario_name)
        
        # Analyze behavior multiple times to simulate progression
        for step in range(3):
            result = detector.analyze_person_behavior(person_data, frame)
            is_suspicious = detector.is_behavior_suspicious(result)
            
            print(f"Step {step + 1}: Stage={result['shoplifting_stage']}, Score={result['suspicious_score']:.2f}, Suspicious={is_suspicious}")
            
            if result['behaviors']:
                for behavior in result['behaviors']:
                    print(f"  - {behavior['type']}: {behavior['description']}")
        
        final_result = detector.analyze_person_behavior(person_data, frame)
        final_suspicious = detector.is_behavior_suspicious(final_result)
        
        results.append({
            'scenario': scenario_name,
            'expected': expected_suspicious,
            'detected': final_suspicious,
            'score': final_result['suspicious_score'],
            'stage': final_result['shoplifting_stage'],
            'correct': expected_suspicious == final_suspicious
        })
        
        print(f"Final: Expected={expected_suspicious}, Detected={final_suspicious}, Correct={expected_suspicious == final_suspicious}")
    
    # Summary
    print(f"\n=== Results Summary ===")
    correct_predictions = sum(1 for r in results if r['correct'])
    total_tests = len(results)
    accuracy = correct_predictions / total_tests
    
    print(f"Accuracy: {accuracy:.2f} ({correct_predictions}/{total_tests})")
    
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"{status} {result['scenario']}: Expected={result['expected']}, Got={result['detected']} (Score: {result['score']:.2f})")
    
    return results

def test_progressive_shoplifting():
    """Test complete shoplifting progression"""
    print("\n=== Testing Progressive Shoplifting ===")
    
    detector = ShopliftingDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    person_id = 100
    
    # Stage 1: Approach shelf
    print("\nStage 1: Approaching shelf...")
    person_data = create_realistic_person_data(person_id, "normal_browsing")
    result1 = detector.analyze_person_behavior(person_data, frame)
    print(f"Result: Stage={result1['shoplifting_stage']}, Score={result1['suspicious_score']:.2f}")
    
    # Stage 2: Interact with shelf
    print("\nStage 2: Interacting with shelf...")
    person_data = create_realistic_person_data(person_id, "shelf_interaction")
    result2 = detector.analyze_person_behavior(person_data, frame)
    print(f"Result: Stage={result2['shoplifting_stage']}, Score={result2['suspicious_score']:.2f}")
    
    # Stage 3: Conceal item
    print("\nStage 3: Concealing item...")
    person_data = create_realistic_person_data(person_id, "concealment_behavior")
    result3 = detector.analyze_person_behavior(person_data, frame)
    print(f"Result: Stage={result3['shoplifting_stage']}, Score={result3['suspicious_score']:.2f}")
    
    # Stage 4: Move toward exit
    print("\nStage 4: Moving toward exit...")
    person_data = create_realistic_person_data(person_id, "exit_movement")
    result4 = detector.analyze_person_behavior(person_data, frame)
    is_suspicious = detector.is_behavior_suspicious(result4)
    print(f"Result: Stage={result4['shoplifting_stage']}, Score={result4['suspicious_score']:.2f}, Suspicious={is_suspicious}")
    
    if result4['behaviors']:
        print("Behaviors detected:")
        for behavior in result4['behaviors']:
            print(f"  - {behavior['type']}: {behavior['description']}")
    
    # Get performance metrics
    metrics = detector.get_performance_metrics()
    print(f"\nDetector Performance:")
    print(f"Total detections: {metrics['total_detections']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"False positive rate: {metrics['false_positive_rate']:.2f}")

if __name__ == "__main__":
    print("Realistic Shoplifting Detection Test")
    print("=" * 50)
    
    test_realistic_scenarios()
    test_progressive_shoplifting()
    
    print("\n" + "=" * 50)
    print("Realistic testing completed!")
