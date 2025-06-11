import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from config import Config


class BehaviorAnalyzer:
    def __init__(self):
        self.crouching_threshold = Config.CROUCHING_THRESHOLD
        self.hand_movement_threshold = Config.HAND_MOVEMENT_THRESHOLD
        self.loitering_threshold = Config.LOITERING_TIME_THRESHOLD

        # Shoplifting detection state tracking
        self.person_states = {}  # Track each person's shoplifting progression
        self.shelf_zones = self._define_shelf_zones()  # Define shelf/rack areas
        self.object_detector = None  # Will be injected from main

    def analyze_person_behavior(self, person_data: Dict, frame: np.ndarray) -> Dict:
        """
        Analyze behavior of a tracked person
        Returns dictionary with behavior analysis results
        """
        person_id = person_data['person_id']
        bbox = person_data['bbox']
        track_data = person_data['track_data']

        behaviors = {
            'person_id': person_id,
            'timestamp': datetime.now(),
            'suspicious_score': 0.0,
            'behaviors': []
        }

        # Analyze crouching behavior
        crouching_score = self._detect_crouching(bbox)
        if crouching_score > 0:
            behaviors['behaviors'].append({
                'type': 'crouching',
                'confidence': crouching_score,
                'description': f'Person appears to be crouching (confidence: {crouching_score:.2f})'
            })
            behaviors['suspicious_score'] += crouching_score * 0.3

        # Analyze loitering behavior
        loitering_score = self._detect_loitering(track_data)
        if loitering_score > 0:
            behaviors['behaviors'].append({
                'type': 'loitering',
                'confidence': loitering_score,
                'description': f'Person has been in area for extended time (confidence: {loitering_score:.2f})'
            })
            behaviors['suspicious_score'] += loitering_score * 0.4

        # Analyze erratic movement
        erratic_score = self._detect_erratic_movement(track_data)
        if erratic_score > 0:
            behaviors['behaviors'].append({
                'type': 'erratic_movement',
                'confidence': erratic_score,
                'description': f'Person showing erratic movement patterns (confidence: {erratic_score:.2f})'
            })
            behaviors['suspicious_score'] += erratic_score * 0.2

        # Analyze hand movements (simplified - would need pose estimation for full implementation)
        hand_movement_score = self._detect_suspicious_hand_movements(
            bbox, track_data)
        if hand_movement_score > 0:
            behaviors['behaviors'].append({
                'type': 'suspicious_hand_movement',
                'confidence': hand_movement_score,
                'description': f'Detected suspicious hand movements (confidence: {hand_movement_score:.2f})'
            })
            behaviors['suspicious_score'] += hand_movement_score * 0.3

        # Analyze proximity to items (if items are detected nearby)
        proximity_score = self._analyze_item_proximity(person_data, frame)
        if proximity_score > 0:
            behaviors['behaviors'].append({
                'type': 'item_proximity',
                'confidence': proximity_score,
                'description': f'Person in close proximity to items (confidence: {proximity_score:.2f})'
            })
            behaviors['suspicious_score'] += proximity_score * 0.2

        # Cap suspicious score at 1.0
        behaviors['suspicious_score'] = min(behaviors['suspicious_score'], 1.0)

        return behaviors

    def _detect_crouching(self, bbox: Dict) -> float:
        """
        Detect if person is crouching based on bounding box aspect ratio
        """
        width = bbox['width']
        height = bbox['height']

        if height == 0:
            return 0.0

        aspect_ratio = width / height

        # Normal standing person has aspect ratio around 0.4-0.6
        # Crouching person has higher aspect ratio (wider relative to height)
        if aspect_ratio > 0.8:
            # Calculate confidence based on how much the ratio deviates from normal
            confidence = min((aspect_ratio - 0.6) / 0.4, 1.0)
            return confidence

        return 0.0

    def _detect_loitering(self, track_data: Dict) -> float:
        """
        Detect if person has been loitering in the same area
        """
        positions = track_data['positions']

        if len(positions) < 10:  # Need sufficient position history
            return 0.0

        # Calculate time spent in area
        first_time = positions[0]['timestamp']
        last_time = positions[-1]['timestamp']
        time_in_area = (last_time - first_time).total_seconds()

        if time_in_area < self.loitering_threshold:
            return 0.0

        # Calculate movement within the area
        recent_positions = positions[-20:]  # Last 20 positions
        center_x = np.mean([pos['x'] for pos in recent_positions])
        center_y = np.mean([pos['y'] for pos in recent_positions])

        # Calculate average distance from center
        distances = [np.sqrt((pos['x'] - center_x)**2 + (pos['y'] - center_y)**2)
                     for pos in recent_positions]
        avg_distance = np.mean(distances)

        # If person is staying in small area for long time, it's suspicious
        if avg_distance < 50 and time_in_area > self.loitering_threshold:
            confidence = min(
                time_in_area / (self.loitering_threshold * 2), 1.0)
            return confidence

        return 0.0

    def _detect_erratic_movement(self, track_data: Dict) -> float:
        """
        Detect erratic or suspicious movement patterns
        """
        positions = track_data['positions']

        if len(positions) < 5:
            return 0.0

        # Calculate direction changes and speed variations
        speeds = []
        direction_changes = 0
        prev_direction = None

        for i in range(1, len(positions)):
            curr_pos = positions[i]
            prev_pos = positions[i-1]

            # Calculate speed
            distance = np.sqrt((curr_pos['x'] - prev_pos['x'])**2 +
                               (curr_pos['y'] - prev_pos['y'])**2)
            time_diff = (curr_pos['timestamp'] -
                         prev_pos['timestamp']).total_seconds()
            if time_diff > 0:
                speed = distance / time_diff
                speeds.append(speed)

            # Calculate direction changes
            if distance > 10:  # Only consider significant movements
                direction = np.arctan2(curr_pos['y'] - prev_pos['y'],
                                       curr_pos['x'] - prev_pos['x'])
                if prev_direction is not None:
                    angle_diff = abs(direction - prev_direction)
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    if angle_diff > np.pi / 3:  # 60 degrees threshold
                        direction_changes += 1
                prev_direction = direction

        if not speeds:
            return 0.0

        # Calculate speed variation
        speed_std = np.std(speeds)
        speed_mean = np.mean(speeds)
        speed_cv = speed_std / max(speed_mean, 1)  # Coefficient of variation

        # High direction changes or high speed variation indicates erratic movement
        direction_score = min(direction_changes / len(positions), 1.0)
        speed_score = min(speed_cv / 2.0, 1.0)

        return max(direction_score, speed_score) * 0.8

    def _detect_suspicious_hand_movements(self, bbox: Dict, track_data: Dict) -> float:
        """
        Simplified hand movement detection based on bounding box changes
        In a full implementation, this would use pose estimation
        """
        positions = track_data['positions']

        if len(positions) < 3:
            return 0.0

        # Look for rapid changes in bounding box dimensions
        # which might indicate hand movements
        recent_positions = positions[-5:]
        width_changes = []

        for i in range(1, len(recent_positions)):
            width_change = abs(
                recent_positions[i]['width'] - recent_positions[i-1]['width'])
            width_changes.append(width_change)

        if width_changes:
            avg_width_change = np.mean(width_changes)
            if avg_width_change > 10:  # Threshold for significant width changes
                confidence = min(avg_width_change / 30, 1.0)
                return confidence * 0.6  # Lower confidence since this is simplified

        return 0.0

    def _analyze_item_proximity(self, person_data: Dict, frame: np.ndarray) -> float:
        """
        Analyze if person is in close proximity to items
        This is a placeholder - would need item detection integration
        """
        # This would integrate with object detector to find nearby items
        # For now, return 0 as placeholder
        return 0.0

    def is_behavior_suspicious(self, behavior_analysis: Dict) -> bool:
        """
        Determine if behavior analysis indicates suspicious activity
        """
        return behavior_analysis['suspicious_score'] >= Config.ALERT_THRESHOLD
