import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from config import Config


class ShopliftingDetector:
    """
    Advanced shoplifting detection system that focuses on actual theft behavior:
    1. Taking items from shelves/racks/displays
    2. Concealing items on person (pockets, bags, clothing)
    3. Intent to leave without paying
    """

    def __init__(self):
        # Shoplifting detection state tracking
        self.person_states = {}  # Track each person's shoplifting progression
        self.shelf_zones = self._define_shelf_zones()  # Define shelf/rack areas
        self.object_detector = None  # Will be injected from main

        # Improved detection thresholds (more sensitive for better detection)
        self.shelf_interaction_threshold = 0.4  # Lowered from 0.6
        self.concealment_threshold = 0.5        # Lowered from 0.7
        self.shoplifting_threshold = 0.7        # Lowered from 0.9

        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'alerts_generated': 0,
            'accuracy_history': []
        }

    def set_object_detector(self, object_detector):
        """Inject object detector for item detection"""
        self.object_detector = object_detector

    def analyze_person_behavior(self, person_data: Dict, frame: np.ndarray) -> Dict:
        """
        Analyze behavior focusing on actual shoplifting detection:
        1. Taking items from shelves/racks
        2. Concealing items on person
        3. Intent to leave without paying
        """
        person_id = person_data['person_id']
        bbox = person_data['bbox']
        track_data = person_data['track_data']

        # Initialize or get person's shoplifting state
        if person_id not in self.person_states:
            self.person_states[person_id] = {
                'stage': 'browsing',  # browsing -> taking -> concealing -> shoplifting
                'items_taken': [],
                'concealment_events': [],
                'shelf_interactions': [],
                'last_update': datetime.now(),
                'suspicious_actions': []
            }

        person_state = self.person_states[person_id]
        person_state['last_update'] = datetime.now()

        behaviors = {
            'person_id': person_id,
            'timestamp': datetime.now(),
            'suspicious_score': 0.0,
            'behaviors': [],
            'shoplifting_stage': person_state['stage']
        }

        # Step 1: Detect shelf/rack interaction (taking items)
        shelf_interaction = self._detect_shelf_interaction(person_data, frame)
        if shelf_interaction['detected']:
            person_state['shelf_interactions'].append(shelf_interaction)
            behaviors['behaviors'].append({
                'type': 'shelf_interaction',
                'confidence': shelf_interaction['confidence'],
                'description': f'Person interacting with shelf/display area'
            })

            # Check if person is taking an item
            item_taken = self._detect_item_taking(
                person_data, frame, shelf_interaction)
            if item_taken['detected']:
                person_state['items_taken'].append(item_taken)
                person_state['stage'] = 'taking'
                behaviors['behaviors'].append({
                    'type': 'item_taking',
                    'confidence': item_taken['confidence'],
                    'description': f'Person appears to be taking item from shelf'
                })
                behaviors['suspicious_score'] += 0.3

        # Step 2: Detect concealment behavior (hiding items)
        if person_state['items_taken']:  # Only check if person has taken items
            concealment = self._detect_concealment_behavior(person_data, frame)
            if concealment['detected']:
                person_state['concealment_events'].append(concealment)
                person_state['stage'] = 'concealing'
                behaviors['behaviors'].append({
                    'type': 'concealment',
                    'confidence': concealment['confidence'],
                    'description': f'Person concealing item: {concealment["method"]}'
                })
                behaviors['suspicious_score'] += 0.5

        # Step 3: Detect shoplifting intent (concealed item + movement toward exit)
        if person_state['stage'] == 'concealing' and person_state['concealment_events']:
            shoplifting_intent = self._detect_shoplifting_intent(
                person_data, frame)
            if shoplifting_intent['detected']:
                person_state['stage'] = 'shoplifting'
                behaviors['behaviors'].append({
                    'type': 'shoplifting_intent',
                    'confidence': shoplifting_intent['confidence'],
                    'description': 'SHOPLIFTING DETECTED: Person with concealed item attempting to leave'
                })
                behaviors['suspicious_score'] = 1.0  # Maximum alert

        # Update behavior stage
        behaviors['shoplifting_stage'] = person_state['stage']

        return behaviors

    def _define_shelf_zones(self) -> List[Dict]:
        """
        Define areas where shelves/racks are located
        In a real implementation, this would be configured per store layout
        """
        # For now, define some example shelf zones
        # These would be configured based on actual store layout
        return [
            {'x1': 50, 'y1': 50, 'x2': 200, 'y2': 400, 'type': 'shelf'},
            {'x1': 250, 'y1': 50, 'x2': 400, 'y2': 400, 'type': 'rack'},
            {'x1': 450, 'y1': 50, 'x2': 600, 'y2': 400, 'type': 'display'},
        ]

    def _detect_shelf_interaction(self, person_data: Dict, frame: np.ndarray) -> Dict:
        """
        Detect if person is interacting with shelves/racks
        """
        bbox = person_data['bbox']
        person_center = (bbox['center_x'], bbox['center_y'])

        # Check if person is near any shelf zones
        for shelf_zone in self.shelf_zones:
            if (shelf_zone['x1'] <= person_center[0] <= shelf_zone['x2'] and
                    shelf_zone['y1'] <= person_center[1] <= shelf_zone['y2']):

                # Person is in shelf area - check for interaction behavior
                interaction_score = self._analyze_shelf_interaction_behavior(
                    person_data, shelf_zone)

                if interaction_score > self.shelf_interaction_threshold:
                    return {
                        'detected': True,
                        'confidence': interaction_score,
                        'shelf_zone': shelf_zone,
                        'timestamp': datetime.now()
                    }

        return {'detected': False, 'confidence': 0.0}

    def _analyze_shelf_interaction_behavior(self, person_data: Dict, shelf_zone: Dict) -> float:
        """
        Analyze specific behaviors that indicate shelf interaction (improved sensitivity)
        """
        track_data = person_data['track_data']
        positions = track_data['positions']

        if len(positions) < 3:  # Reduced from 5 for faster detection
            return 0.0

        score = 0.0

        # Check if person has been stationary near shelf (reaching/grabbing behavior)
        recent_positions = positions[-min(10, len(positions)):]
        movement_distances = []

        for i in range(1, len(recent_positions)):
            dist = np.sqrt((recent_positions[i]['x'] - recent_positions[i-1]['x'])**2 +
                           (recent_positions[i]['y'] - recent_positions[i-1]['y'])**2)
            movement_distances.append(dist)

        if movement_distances:
            avg_movement = np.mean(movement_distances)
            # Improved stationary detection (more sensitive)
            if avg_movement < 25:  # Increased from 20 for better detection
                score += 0.5  # Increased from 0.4

            # Add time-based scoring - longer stationary = higher score
            if len(recent_positions) >= 8 and avg_movement < 15:
                score += 0.3

        # Check for hand-reaching behavior (improved sensitivity)
        bbox_changes = self._analyze_bbox_changes(
            positions[-min(5, len(positions)):])
        if bbox_changes > 0.2:  # Lowered from 0.3 for better sensitivity
            score += 0.4  # Increased from 0.3

        # Add proximity scoring - closer to shelf edge = higher score
        bbox = person_data['bbox']
        shelf_proximity = self._calculate_shelf_proximity(bbox, shelf_zone)
        score += shelf_proximity * 0.3

        return min(score, 1.0)

    def _detect_item_taking(self, person_data: Dict, frame: np.ndarray, shelf_interaction: Dict) -> Dict:
        """
        Detect if person is actually taking an item from shelf
        """
        if not self.object_detector:
            # Fallback to behavioral analysis
            return self._detect_item_taking_behavioral(person_data, shelf_interaction)

        # Use object detection to identify items being taken
        return self._detect_item_taking_with_objects(person_data, frame, shelf_interaction)

    def _detect_item_taking_behavioral(self, person_data: Dict, shelf_interaction: Dict) -> Dict:
        """
        Behavioral detection of item taking (improved sensitivity)
        """
        track_data = person_data['track_data']
        positions = track_data['positions']

        if len(positions) < 2:  # Reduced from 3 for faster detection
            return {'detected': False, 'confidence': 0.0}

        # Look for specific movement patterns that indicate taking items
        # 1. Approach shelf
        # 2. Stop/slow down
        # 3. Reach (bbox changes)
        # 4. Retract (return to normal position)

        score = 0.0

        # Check for approach-stop-reach pattern (improved)
        if len(positions) >= 6:  # Reduced from 10
            mid_point = len(positions) // 2
            early_positions = positions[:mid_point]
            recent_positions = positions[mid_point:]

            # Calculate movement in early vs recent positions
            early_movement = self._calculate_average_movement(early_positions)
            recent_movement = self._calculate_average_movement(
                recent_positions)

            # If person was moving then stopped, it indicates reaching behavior
            if early_movement > 20 and recent_movement < 20:  # More sensitive thresholds
                score += 0.6  # Increased from 0.5

        # Check for bbox changes indicating reaching (improved)
        bbox_changes = self._analyze_bbox_changes(
            positions[-min(3, len(positions)):])
        if bbox_changes > 0.25:  # Lowered from 0.4 for better sensitivity
            score += 0.5  # Increased from 0.4

        # Add time-based scoring for sustained interaction
        if len(positions) >= 5:
            score += 0.2

        # Add shelf interaction confidence boost
        shelf_confidence = shelf_interaction.get('confidence', 0.0)
        score += shelf_confidence * 0.3

        confidence = min(score, 1.0)

        return {
            'detected': confidence > 0.4,  # Lowered from 0.6 for better detection
            'confidence': confidence,
            'method': 'behavioral_analysis',
            'timestamp': datetime.now()
        }

    def _detect_item_taking_with_objects(self, person_data: Dict, frame: np.ndarray, shelf_interaction: Dict) -> Dict:
        """
        Object-detection based item taking detection
        """
        # Detect objects in the frame
        objects = self.object_detector.detect_items(frame)

        bbox = person_data['bbox']
        person_area = {
            'x1': bbox['x1'], 'y1': bbox['y1'],
            'x2': bbox['x2'], 'y2': bbox['y2']
        }

        # Check for objects that are very close to or overlapping with person
        for obj in objects:
            obj_bbox = obj['bbox']

            # Calculate overlap or proximity
            overlap = self._calculate_overlap(person_area, obj_bbox)
            proximity = self._calculate_proximity(person_area, obj_bbox)

            if overlap > 0.1 or proximity < 30:  # Object very close to person
                return {
                    'detected': True,
                    'confidence': 0.8,
                    'method': 'object_detection',
                    'item_type': obj['class'],
                    'timestamp': datetime.now()
                }

        return {'detected': False, 'confidence': 0.0}

    def _detect_concealment_behavior(self, person_data: Dict, frame: np.ndarray) -> Dict:
        """
        Detect if person is concealing items (putting in pockets, bags, etc.)
        """
        bbox = person_data['bbox']
        track_data = person_data['track_data']
        positions = track_data['positions']

        if len(positions) < 5:
            return {'detected': False, 'confidence': 0.0}

        concealment_score = 0.0
        concealment_method = "unknown"

        # Method 1: Detect turning away from camera (hiding action)
        turning_score = self._detect_turning_away(person_data)
        if turning_score > 0.5:
            concealment_score += 0.4
            concealment_method = "turning_away"

        # Method 2: Detect crouching/bending (hiding behind body)
        crouching_score = self._detect_concealment_crouching(bbox)
        if crouching_score > 0.5:
            concealment_score += 0.3
            concealment_method = "crouching"

        # Method 3: Detect hand-to-body movements (putting in pockets/jacket)
        hand_to_body_score = self._detect_hand_to_body_movement(person_data)
        if hand_to_body_score > 0.5:
            concealment_score += 0.5
            concealment_method = "pocket_concealment"

        # Method 4: Detect bag/container interaction
        bag_interaction_score = self._detect_bag_interaction(
            person_data, frame)
        if bag_interaction_score > 0.5:
            concealment_score += 0.6
            concealment_method = "bag_concealment"

        confidence = min(concealment_score, 1.0)

        return {
            'detected': confidence > self.concealment_threshold,
            'confidence': confidence,
            'method': concealment_method,
            'timestamp': datetime.now()
        }

    def _detect_turning_away(self, person_data: Dict) -> float:
        """
        Detect if person is turning away from camera (suspicious behavior)
        """
        bbox = person_data['bbox']
        track_data = person_data['track_data']
        positions = track_data['positions']

        if len(positions) < 5:
            return 0.0

        # Analyze bbox width changes - turning away typically reduces visible width
        recent_positions = positions[-5:]
        width_changes = []

        for i in range(1, len(recent_positions)):
            width_change = (
                recent_positions[i-1]['width'] - recent_positions[i]['width']) / recent_positions[i-1]['width']
            width_changes.append(width_change)

        # If width is consistently decreasing, person might be turning away
        if width_changes:
            avg_width_change = np.mean(width_changes)
            if avg_width_change > 0.1:  # 10% width reduction
                return min(avg_width_change * 3, 1.0)

        return 0.0

    def _detect_concealment_crouching(self, bbox: Dict) -> float:
        """
        Detect crouching specifically for concealment (different from general crouching)
        """
        width = bbox['width']
        height = bbox['height']

        if height == 0:
            return 0.0

        aspect_ratio = width / height

        # Concealment crouching is more extreme than casual crouching
        if aspect_ratio > 1.0:  # Very wide relative to height
            confidence = min((aspect_ratio - 0.8) / 0.5, 1.0)
            return confidence

        return 0.0

    def _detect_hand_to_body_movement(self, person_data: Dict) -> float:
        """
        Detect hand movements toward body (putting items in pockets, jacket, etc.)
        """
        track_data = person_data['track_data']
        positions = track_data['positions']

        if len(positions) < 3:
            return 0.0

        # Look for specific bbox changes that indicate hand-to-body movement
        recent_positions = positions[-3:]

        # Check for width expansion followed by contraction (reaching then concealing)
        if len(recent_positions) >= 3:
            width1 = recent_positions[0]['width']
            width2 = recent_positions[1]['width']
            width3 = recent_positions[2]['width']

            # Pattern: normal -> expanded (reaching) -> contracted (concealing)
            if width2 > width1 * 1.1 and width3 < width2 * 0.9:
                return 0.7

        return 0.0

    def _detect_bag_interaction(self, person_data: Dict, frame: np.ndarray) -> float:
        """
        Detect interaction with bags/containers for concealment
        """
        if not self.object_detector:
            return 0.0

        # Detect bags/containers in the frame
        objects = self.object_detector.detect_objects(frame)
        bags = [obj for obj in objects if obj['class']
                in ['handbag', 'backpack', 'suitcase']]

        if not bags:
            return 0.0

        bbox = person_data['bbox']
        person_area = {
            'x1': bbox['x1'], 'y1': bbox['y1'],
            'x2': bbox['x2'], 'y2': bbox['y2']
        }

        # Check if person is interacting with any bags
        for bag in bags:
            bag_bbox = bag['bbox']
            overlap = self._calculate_overlap(person_area, bag_bbox)

            if overlap > 0.2:  # Significant overlap indicates interaction
                return 0.8

        return 0.0

    def _detect_shoplifting_intent(self, person_data: Dict, frame: np.ndarray) -> Dict:
        """
        Detect intent to leave with concealed items (final shoplifting confirmation)
        """
        track_data = person_data['track_data']
        positions = track_data['positions']

        if len(positions) < 10:
            return {'detected': False, 'confidence': 0.0}

        intent_score = 0.0

        # Check movement toward exit areas
        exit_movement_score = self._detect_exit_movement(positions)
        if exit_movement_score > 0.5:
            intent_score += 0.5

        # Check for avoiding staff/cameras
        avoidance_score = self._detect_avoidance_behavior(person_data)
        if avoidance_score > 0.5:
            intent_score += 0.3

        # Check for quick/nervous movement patterns
        nervous_movement_score = self._detect_nervous_movement(positions)
        if nervous_movement_score > 0.5:
            intent_score += 0.2

        confidence = min(intent_score, 1.0)

        return {
            'detected': confidence > self.shoplifting_threshold,
            'confidence': confidence,
            'timestamp': datetime.now()
        }

    # Utility methods
    def _analyze_bbox_changes(self, positions: List[Dict]) -> float:
        """Analyze bounding box changes to detect reaching/movement"""
        if len(positions) < 2:
            return 0.0

        width_changes = []
        height_changes = []

        for i in range(1, len(positions)):
            width_change = abs(
                positions[i]['width'] - positions[i-1]['width']) / positions[i-1]['width']
            height_change = abs(
                positions[i]['height'] - positions[i-1]['height']) / positions[i-1]['height']
            width_changes.append(width_change)
            height_changes.append(height_change)

        avg_width_change = np.mean(width_changes) if width_changes else 0
        avg_height_change = np.mean(height_changes) if height_changes else 0

        return max(avg_width_change, avg_height_change)

    def _calculate_average_movement(self, positions: List[Dict]) -> float:
        """Calculate average movement distance between positions"""
        if len(positions) < 2:
            return 0.0

        distances = []
        for i in range(1, len(positions)):
            dist = np.sqrt((positions[i]['x'] - positions[i-1]['x'])**2 +
                           (positions[i]['y'] - positions[i-1]['y'])**2)
            distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def _calculate_shelf_proximity(self, bbox: Dict, shelf_zone: Dict) -> float:
        """Calculate how close person is to shelf edge (0.0 to 1.0)"""
        person_center_x = bbox['center_x']
        person_center_y = bbox['center_y']

        # Calculate distance to nearest shelf edge
        shelf_center_x = (shelf_zone['x1'] + shelf_zone['x2']) / 2
        shelf_center_y = (shelf_zone['y1'] + shelf_zone['y2']) / 2

        distance = np.sqrt((person_center_x - shelf_center_x)**2 +
                           (person_center_y - shelf_center_y)**2)

        # Convert to proximity score (closer = higher score)
        max_distance = 100  # Maximum meaningful distance
        proximity = max(0, 1 - (distance / max_distance))
        return proximity

    def _calculate_overlap(self, area1: Dict, area2: Dict) -> float:
        """Calculate overlap ratio between two rectangular areas"""
        x1 = max(area1['x1'], area2['x1'])
        y1 = max(area1['y1'], area2['y1'])
        x2 = min(area1['x2'], area2['x2'])
        y2 = min(area1['y2'], area2['y2'])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        overlap_area = (x2 - x1) * (y2 - y1)
        area1_size = (area1['x2'] - area1['x1']) * (area1['y2'] - area1['y1'])
        area2_size = (area2['x2'] - area2['x1']) * (area2['y2'] - area2['y1'])

        return overlap_area / min(area1_size, area2_size)

    def _calculate_proximity(self, area1: Dict, area2: Dict) -> float:
        """Calculate distance between centers of two areas"""
        center1_x = (area1['x1'] + area1['x2']) / 2
        center1_y = (area1['y1'] + area1['y2']) / 2
        center2_x = (area2['x1'] + area2['x2']) / 2
        center2_y = (area2['y1'] + area2['y2']) / 2

        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

    def _detect_exit_movement(self, positions: List[Dict]) -> float:
        """Detect movement toward store exits"""
        if len(positions) < 5:
            return 0.0

        # Analyze movement direction over recent positions
        recent_positions = positions[-10:]

        # Calculate general movement direction
        start_pos = recent_positions[0]
        end_pos = recent_positions[-1]

        movement_vector = (end_pos['x'] - start_pos['x'],
                           end_pos['y'] - start_pos['y'])
        movement_distance = np.sqrt(
            movement_vector[0]**2 + movement_vector[1]**2)

        if movement_distance < 50:  # Not moving much
            return 0.0

        # Define exit areas (would be configured per store)
        exit_areas = [
            {'x': 0, 'y': 300, 'radius': 100},  # Left exit
            {'x': 640, 'y': 300, 'radius': 100},  # Right exit
        ]

        # Check if movement is toward any exit
        for exit_area in exit_areas:
            distance_to_exit = np.sqrt((end_pos['x'] - exit_area['x'])**2 +
                                       (end_pos['y'] - exit_area['y'])**2)

            if distance_to_exit < exit_area['radius']:
                # Calculate how directly person is moving toward exit
                exit_vector = (
                    exit_area['x'] - start_pos['x'], exit_area['y'] - start_pos['y'])

                # Dot product to measure alignment
                dot_product = (movement_vector[0] * exit_vector[0] +
                               movement_vector[1] * exit_vector[1])

                movement_magnitude = np.sqrt(
                    movement_vector[0]**2 + movement_vector[1]**2)
                exit_magnitude = np.sqrt(exit_vector[0]**2 + exit_vector[1]**2)

                if movement_magnitude > 0 and exit_magnitude > 0:
                    alignment = dot_product / \
                        (movement_magnitude * exit_magnitude)
                    if alignment > 0.5:  # Moving toward exit
                        return min(alignment, 1.0)

        return 0.0

    def _detect_avoidance_behavior(self, person_data: Dict) -> float:
        """Detect if person is avoiding cameras/staff (simplified)"""
        # This would be more sophisticated in a real implementation
        # For now, return low score as placeholder
        return 0.0

    def _detect_nervous_movement(self, positions: List[Dict]) -> float:
        """Detect nervous/erratic movement patterns"""
        if len(positions) < 5:
            return 0.0

        # Calculate speed variations (nervous people move erratically)
        speeds = []
        for i in range(1, len(positions)):
            dist = np.sqrt((positions[i]['x'] - positions[i-1]['x'])**2 +
                           (positions[i]['y'] - positions[i-1]['y'])**2)
            time_diff = (positions[i]['timestamp'] -
                         positions[i-1]['timestamp']).total_seconds()
            if time_diff > 0:
                speeds.append(dist / time_diff)

        if len(speeds) < 3:
            return 0.0

        # High speed variation indicates nervous movement
        speed_std = np.std(speeds)
        speed_mean = np.mean(speeds)

        if speed_mean > 0:
            coefficient_of_variation = speed_std / speed_mean
            return min(coefficient_of_variation / 2.0, 1.0)

        return 0.0

    def update_detection_stats(self, is_true_positive: bool, alert_generated: bool):
        """Update performance statistics"""
        self.detection_stats['total_detections'] += 1
        if is_true_positive:
            self.detection_stats['true_positives'] += 1
        else:
            self.detection_stats['false_positives'] += 1

        if alert_generated:
            self.detection_stats['alerts_generated'] += 1

        # Calculate current accuracy
        total = self.detection_stats['total_detections']
        if total > 0:
            accuracy = self.detection_stats['true_positives'] / total
            self.detection_stats['accuracy_history'].append(accuracy)

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        total = self.detection_stats['total_detections']
        if total == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'false_positive_rate': 0.0,
                'total_detections': 0,
                'alerts_generated': 0
            }

        tp = self.detection_stats['true_positives']
        fp = self.detection_stats['false_positives']

        accuracy = tp / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        false_positive_rate = fp / total if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'false_positive_rate': false_positive_rate,
            'total_detections': total,
            'true_positives': tp,
            'false_positives': fp,
            'alerts_generated': self.detection_stats['alerts_generated'],
            'avg_accuracy': np.mean(self.detection_stats['accuracy_history']) if self.detection_stats['accuracy_history'] else 0.0
        }

    def is_behavior_suspicious(self, behavior_analysis: Dict) -> bool:
        """
        Determine if behavior analysis indicates actual shoplifting
        Only trigger alerts for confirmed shoplifting behavior
        """
        # Only alert if we've detected actual shoplifting (stage progression)
        stage = behavior_analysis.get('shoplifting_stage', 'browsing')
        suspicious_score = behavior_analysis.get('suspicious_score', 0.0)

        # Alert only for concealing or shoplifting stages with high confidence
        return (stage in ['concealing', 'shoplifting'] and suspicious_score >= 0.7)

    def cleanup_old_states(self, max_age_minutes: int = 30):
        """Clean up old person states to prevent memory leaks"""
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)

        expired_ids = []
        for person_id, state in self.person_states.items():
            if state['last_update'] < cutoff_time:
                expired_ids.append(person_id)

        for person_id in expired_ids:
            del self.person_states[person_id]
