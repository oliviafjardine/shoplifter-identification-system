#!/usr/bin/env python3
"""
Test script to verify person tracking behavior
"""

import sys
import time
from datetime import datetime
from detection.tracker import PersonTracker

def test_track_clearing():
    """Test that tracks are cleared properly when no people are detected"""
    print("Testing track clearing behavior...")
    
    tracker = PersonTracker()
    
    # Simulate a person detection
    fake_detection = {
        'class': 'person',
        'confidence': 0.8,
        'bbox': {
            'x1': 100,
            'y1': 100,
            'x2': 200,
            'y2': 300,
            'width': 100,
            'height': 200,
            'center_x': 150,
            'center_y': 200
        }
    }
    
    print("1. Adding a person detection...")
    tracked_people = tracker.update_tracks([fake_detection])
    print(f"   Active tracks: {len(tracker.tracks)}")
    print(f"   Tracked people: {len(tracked_people)}")
    
    # Wait a moment
    time.sleep(0.1)
    
    print("2. First frame with no detections...")
    tracked_people = tracker.update_tracks([])
    print(f"   Active tracks: {len(tracker.tracks)}")
    print(f"   Tracked people: {len(tracked_people)}")
    
    print("3. Second frame with no detections...")
    tracked_people = tracker.update_tracks([])
    print(f"   Active tracks: {len(tracker.tracks)}")
    print(f"   Tracked people: {len(tracked_people)}")
    
    print("4. Third frame with no detections...")
    tracked_people = tracker.update_tracks([])
    print(f"   Active tracks: {len(tracker.tracks)}")
    print(f"   Tracked people: {len(tracked_people)}")
    
    # Test should show tracks being cleared after 2 frames with no detections
    if len(tracker.tracks) == 0:
        print("✅ SUCCESS: Tracks cleared properly!")
    else:
        print("❌ FAILURE: Tracks not cleared!")
    
    return len(tracker.tracks) == 0

def test_timeout_clearing():
    """Test that tracks are cleared after timeout"""
    print("\nTesting timeout-based track clearing...")
    
    tracker = PersonTracker()
    
    # Simulate a person detection
    fake_detection = {
        'class': 'person',
        'confidence': 0.8,
        'bbox': {
            'x1': 100,
            'y1': 100,
            'x2': 200,
            'y2': 300,
            'width': 100,
            'height': 200,
            'center_x': 150,
            'center_y': 200
        }
    }
    
    print("1. Adding a person detection...")
    tracked_people = tracker.update_tracks([fake_detection])
    print(f"   Active tracks: {len(tracker.tracks)}")
    
    # Wait longer than timeout (1.5 seconds)
    print("2. Waiting 1.5 seconds (longer than timeout)...")
    time.sleep(1.5)
    
    print("3. Processing frame after timeout...")
    tracked_people = tracker.update_tracks([])
    print(f"   Active tracks: {len(tracker.tracks)}")
    print(f"   Tracked people: {len(tracked_people)}")
    
    if len(tracker.tracks) == 0:
        print("✅ SUCCESS: Tracks cleared after timeout!")
    else:
        print("❌ FAILURE: Tracks not cleared after timeout!")
    
    return len(tracker.tracks) == 0

if __name__ == "__main__":
    print("Person Tracker Test Suite")
    print("=" * 40)
    
    test1_passed = test_track_clearing()
    test2_passed = test_timeout_clearing()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Track clearing test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Timeout clearing test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)
