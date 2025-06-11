# Shoplifting Detection System Guide

## Overview

The system has been completely redesigned to focus on **actual shoplifting behavior** rather than general suspicious activities. Alerts are now only triggered when the system detects a complete shoplifting sequence: **taking items from shelves → concealing them → attempting to leave**.

## Key Changes

### ❌ What No Longer Triggers Alerts

- **General suspicious behavior** (loitering, erratic movement, crouching)
- **Person detection alone** (just being in the store)
- **Movement patterns** (unless related to theft sequence)
- **Proximity to items** (unless actually taking them)

### ✅ What Now Triggers Alerts

**Only actual shoplifting behavior in this sequence:**

1. **Shelf Interaction** → Person approaches and interacts with merchandise displays
2. **Item Taking** → Person removes item from shelf/rack/display
3. **Concealment** → Person hides item on their person (pocket, jacket, bag, etc.)
4. **Shoplifting Intent** → Person with concealed item moves toward exit

## Detection Stages

### 🔍 Stage 1: Browsing
- **Status**: Normal customer behavior
- **Action**: No alerts
- **Description**: Person walking around, looking at items normally

### 📍 Stage 2: Shelf Interaction  
- **Status**: Monitoring
- **Action**: Low-priority notification
- **Description**: Person interacting with merchandise displays
- **Alert**: "📍 SHELF INTERACTION - Person is interacting with merchandise display"

### 👀 Stage 3: Taking
- **Status**: Item removal detected
- **Action**: Medium-priority alert
- **Description**: Person has taken item from shelf/display
- **Alert**: "👀 ITEM REMOVAL - Person has taken item from shelf/display - Monitoring for concealment behavior"

### ⚠️ Stage 4: Concealing
- **Status**: Concealment detected
- **Action**: High-priority alert
- **Description**: Person hiding merchandise on their person
- **Alert**: "⚠️ CONCEALMENT DETECTED - Person is hiding merchandise on their person - Monitor closely - potential theft in progress"

### 🚨 Stage 5: Shoplifting
- **Status**: CRITICAL - Theft in progress
- **Action**: Critical alert - immediate action required
- **Description**: Person with concealed item attempting to leave
- **Alert**: "🚨 SHOPLIFTING ALERT - CRITICAL: Person has concealed merchandise and is attempting to leave without payment - IMMEDIATE ACTION REQUIRED"

## Detection Methods

### Shelf Interaction Detection
- **Zone-based detection**: Monitors predefined shelf/rack areas
- **Behavioral analysis**: Detects stationary behavior near merchandise
- **Movement patterns**: Identifies approach-stop-reach sequences

### Item Taking Detection
- **Behavioral analysis**: Detects reach-and-retract movement patterns
- **Bounding box changes**: Monitors for reaching gestures
- **Object detection**: (When available) Tracks items near person

### Concealment Detection
- **Turning away**: Detects person turning away from camera
- **Body language**: Identifies crouching/bending to hide actions  
- **Hand-to-body movement**: Detects putting items in pockets/jacket
- **Bag interaction**: Monitors interaction with bags/containers

### Exit Intent Detection
- **Movement tracking**: Analyzes movement toward store exits
- **Avoidance behavior**: Detects attempts to avoid staff/cameras
- **Nervous movement**: Identifies erratic movement patterns

## Configuration

### Shelf Zones
Configure shelf/rack locations in `detection/shoplifting_detector.py`:

```python
def _define_shelf_zones(self):
    return [
        {'x1': 50, 'y1': 50, 'x2': 200, 'y2': 400, 'type': 'shelf'},
        {'x1': 250, 'y1': 50, 'x2': 400, 'y2': 400, 'type': 'rack'},
        # Add more zones based on your store layout
    ]
```

### Detection Thresholds
Adjust sensitivity in `ShopliftingDetector.__init__()`:

```python
self.shelf_interaction_threshold = 0.6    # Lower = more sensitive
self.concealment_threshold = 0.7          # Higher = fewer false positives  
self.shoplifting_threshold = 0.9          # Very high for final confirmation
```

### Exit Areas
Configure store exits in `_detect_exit_movement()`:

```python
exit_areas = [
    {'x': 0, 'y': 300, 'radius': 100},     # Left exit
    {'x': 640, 'y': 300, 'radius': 100},   # Right exit
    # Add more exits as needed
]
```

## Alert Severity Levels

- **📍 Low**: Shelf interaction only
- **👀 Medium**: Item taking detected
- **⚠️ High**: Concealment detected  
- **🚨 Critical**: Shoplifting confirmed

## Benefits of New System

### ✅ Advantages
- **Eliminates false positives** from normal customer behavior
- **Focuses on actual theft** rather than suspicious activities
- **Provides clear progression** from browsing to shoplifting
- **Actionable alerts** with specific stages and recommendations
- **Professional appearance** suitable for real store environments

### 🎯 Use Cases
- **Retail loss prevention** in stores, supermarkets, pharmacies
- **High-value merchandise** protection (electronics, jewelry, etc.)
- **Self-checkout monitoring** for concealment behavior
- **Staff training** on shoplifting recognition patterns

## Testing

Run the test script to see the system in action:

```bash
python test_shoplifting_detection.py
```

This demonstrates:
- Normal browsing (no alerts)
- Shelf interaction only (no alerts)  
- Complete shoplifting sequence (progressive alerts)
- False positive scenarios (no alerts)

## Technical Implementation

### Files Modified
- `detection/shoplifting_detector.py` - New comprehensive shoplifting detector
- `main.py` - Updated to use new detector
- `services/alert_service.py` - Enhanced alert messages
- `static/js/app.js` - Fixed timestamp display

### Key Classes
- `ShopliftingDetector` - Main detection logic
- `AlertService` - Enhanced with shoplifting-specific messages

### State Tracking
Each person is tracked through shoplifting stages with persistent state management and automatic cleanup of old tracking data.

## Conclusion

The system now provides **professional-grade shoplifting detection** that focuses on actual theft behavior rather than generating false alarms from normal customer activities. This makes it suitable for real-world retail environments where accuracy and reliability are essential.
