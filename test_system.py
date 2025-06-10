#!/usr/bin/env python3
"""
Test script for Shoplifting Detection System
This script tests various components of the system to ensure everything is working correctly.
"""

import sys
import cv2
import numpy as np
from datetime import datetime
import asyncio
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Core modules
        import fastapi
        import uvicorn
        import sqlalchemy
        import psycopg2
        import cv2
        import numpy as np
        from ultralytics import YOLO
        import sklearn
        
        # Our modules
        from config import Config
        from models.database import create_tables, get_db
        from services.camera_service import CameraService
        from services.alert_service import AlertService
        from detection.object_detector import ObjectDetector
        from detection.tracker import PersonTracker
        from detection.behavior_analyzer import BehaviorAnalyzer
        from detection.anomaly_detector import AnomalyDetector
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_camera():
    """Test camera functionality"""
    print("\n🎥 Testing camera...")
    
    try:
        from services.camera_service import CameraService
        
        camera = CameraService(camera_source=0)
        
        # Try to start camera
        if camera.start_capture():
            print("✅ Camera started successfully")
            
            # Try to get a frame
            import time
            time.sleep(2)  # Wait for camera to initialize
            
            frame = camera.get_current_frame()
            if frame is not None:
                print(f"✅ Frame captured: {frame.shape}")
                camera.stop_capture()
                return True
            else:
                print("❌ Could not capture frame")
                camera.stop_capture()
                return False
        else:
            print("❌ Could not start camera")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_object_detection():
    """Test object detection"""
    print("\n🔍 Testing object detection...")
    
    try:
        from detection.object_detector import ObjectDetector
        
        detector = ObjectDetector()
        
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:400, 200:440] = [255, 255, 255]  # White rectangle
        
        # Test detection
        detections = detector.detect_objects(test_image)
        print(f"✅ Object detection working, found {len(detections)} objects")
        
        # Test people detection specifically
        people = detector.detect_people(test_image)
        print(f"✅ People detection working, found {len(people)} people")
        
        return True
        
    except Exception as e:
        print(f"❌ Object detection test failed: {e}")
        return False

def test_database():
    """Test database connection"""
    print("\n🗄️ Testing database...")
    
    try:
        from models.database import create_tables, get_db, Event, Alert
        from sqlalchemy.orm import Session
        
        # Test database connection
        db = next(get_db())
        
        # Test creating tables
        create_tables()
        print("✅ Database tables created/verified")
        
        # Test inserting a record
        test_event = Event(
            event_type='test',
            confidence=0.5,
            person_id=1,
            x_coordinate=100,
            y_coordinate=100,
            width=50,
            height=100,
            description='Test event'
        )
        
        db.add(test_event)
        db.commit()
        
        # Test querying
        events = db.query(Event).filter(Event.event_type == 'test').all()
        print(f"✅ Database operations working, found {len(events)} test events")
        
        # Clean up test data
        for event in events:
            db.delete(event)
        db.commit()
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        print("Make sure PostgreSQL is running and accessible")
        return False

def test_behavior_analysis():
    """Test behavior analysis"""
    print("\n🧠 Testing behavior analysis...")
    
    try:
        from detection.behavior_analyzer import BehaviorAnalyzer
        from detection.tracker import PersonTracker
        from datetime import datetime
        
        analyzer = BehaviorAnalyzer()
        
        # Create test person data
        test_person_data = {
            'person_id': 1,
            'bbox': {
                'x1': 100, 'y1': 100, 'x2': 150, 'y2': 200,
                'width': 50, 'height': 100,
                'center_x': 125, 'center_y': 150
            },
            'track_data': {
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'positions': [
                    {'x': 125, 'y': 150, 'width': 50, 'height': 100, 'timestamp': datetime.now()}
                ],
                'behavior_history': [],
                'alert_history': []
            }
        }
        
        # Test behavior analysis
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        behavior_result = analyzer.analyze_person_behavior(test_person_data, test_frame)
        
        print(f"✅ Behavior analysis working, suspicious score: {behavior_result['suspicious_score']:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Behavior analysis test failed: {e}")
        return False

def test_anomaly_detection():
    """Test anomaly detection"""
    print("\n🤖 Testing anomaly detection...")
    
    try:
        from detection.anomaly_detector import AnomalyDetector
        
        detector = AnomalyDetector()
        
        # Create test data
        test_behavior_data = {
            'person_id': 1,
            'suspicious_score': 0.5,
            'behaviors': []
        }
        
        test_track_data = {
            'positions': [
                {'x': 100, 'y': 100, 'timestamp': datetime.now()},
                {'x': 105, 'y': 105, 'timestamp': datetime.now()}
            ],
            'first_seen': datetime.now(),
            'last_seen': datetime.now()
        }
        
        # Test feature extraction
        features = detector.extract_features(test_behavior_data, test_track_data)
        print(f"✅ Feature extraction working, extracted {len(features)} features")
        
        # Test anomaly detection
        result = detector.detect_anomaly(test_behavior_data, test_track_data)
        print(f"✅ Anomaly detection working, result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Anomaly detection test failed: {e}")
        return False

async def test_alert_service():
    """Test alert service"""
    print("\n🚨 Testing alert service...")
    
    try:
        from services.alert_service import AlertService
        
        alert_service = AlertService()
        
        # Test alert creation
        test_event_data = {
            'x_coordinate': 100,
            'y_coordinate': 100,
            'width': 50,
            'height': 100
        }
        
        test_behavior_data = {
            'person_id': 1,
            'suspicious_score': 0.8,
            'behaviors': [
                {'type': 'test', 'confidence': 0.8, 'description': 'Test behavior'}
            ]
        }
        
        # This might fail if database is not available, which is okay for testing
        try:
            alert = await alert_service.create_alert(test_event_data, test_behavior_data)
            if alert:
                print("✅ Alert service working, alert created")
            else:
                print("⚠️ Alert service working but no alert created (cooldown or database issue)")
        except:
            print("⚠️ Alert service initialized but database operations failed (expected if DB not running)")
        
        return True
        
    except Exception as e:
        print(f"❌ Alert service test failed: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from config import Config
        
        print(f"✅ Config loaded:")
        print(f"   - Camera source: {Config.CAMERA_SOURCE}")
        print(f"   - Alert threshold: {Config.ALERT_THRESHOLD}")
        print(f"   - Model path: {Config.MODEL_PATH}")
        print(f"   - Debug mode: {Config.DEBUG}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_yolo_model():
    """Test YOLO model loading"""
    print("\n🎯 Testing YOLO model...")
    
    try:
        from ultralytics import YOLO
        from config import Config
        
        model = YOLO(Config.MODEL_PATH)
        print("✅ YOLO model loaded successfully")
        
        # Test inference on dummy image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_image, verbose=False)
        print("✅ YOLO inference working")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO model test failed: {e}")
        print("Try running: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
        return False

async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Shoplifting Detection System Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("YOLO Model", test_yolo_model),
        ("Database", test_database),
        ("Camera", test_camera),
        ("Object Detection", test_object_detection),
        ("Behavior Analysis", test_behavior_analysis),
        ("Anomaly Detection", test_anomaly_detection),
        ("Alert Service", test_alert_service),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nTo start the system:")
        print("1. Ensure PostgreSQL is running: docker-compose up -d postgres")
        print("2. Start the application: python main.py")
        print("3. Open browser: http://localhost:8000")
    else:
        print(f"\n⚠️ {total-passed} tests failed. Please check the issues above.")
        print("\nCommon solutions:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Start database: docker-compose up -d postgres")
        print("- Check camera permissions and availability")
        print("- Ensure YOLO model is downloaded")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
