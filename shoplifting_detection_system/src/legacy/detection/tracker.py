import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from config import Config
import cv2


class PersonTracker:
    def __init__(self):
        self.tracks = {}  # person_id -> track_data
        self.next_id = 1
        self.max_distance = Config.MAX_TRACKING_DISTANCE
        self.track_timeout = 1.0  # Very aggressive timeout - 1 second for immediate cleanup
        self.frames_without_detection = 0  # Counter for frames without any detections

    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """
        Update person tracks with new detections
        Returns list of tracked persons with IDs
        """
        current_time = datetime.now()
        tracked_people = []

        # Apply Non-Maximum Suppression to remove overlapping detections
        filtered_detections = self._apply_nms(detections)

        # Always remove expired tracks first
        self._remove_expired_tracks(current_time)

        # If no people detected, increment counter and clear tracks aggressively
        if not filtered_detections:
            self.frames_without_detection += 1
            # Clear tracks immediately if no detections for 2 consecutive frames
            if self.frames_without_detection >= 2:
                if self.tracks:
                    print(
                        f"DEBUG: Clearing {len(self.tracks)} tracks - no detections for {self.frames_without_detection} frames")
                self.tracks.clear()
            return []
        else:
            # Reset counter when we have detections
            self.frames_without_detection = 0

        # Match detections to existing tracks
        unmatched_detections = []

        for detection in filtered_detections:
            if detection['class'] != 'person':
                continue

            center_x = detection['bbox']['center_x']
            center_y = detection['bbox']['center_y']

            # Find closest existing track
            best_match_id = None
            best_distance = float('inf')

            for person_id, track_data in self.tracks.items():
                last_pos = track_data['positions'][-1]
                distance = np.sqrt(
                    (center_x - last_pos['x'])**2 + (center_y - last_pos['y'])**2)

                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match_id = person_id

            if best_match_id is not None:
                # Update existing track
                self._update_track(best_match_id, detection, current_time)
                tracked_person = detection.copy()
                tracked_person['person_id'] = best_match_id
                tracked_person['track_data'] = self.tracks[best_match_id]
                tracked_people.append(tracked_person)
            else:
                # Create new track
                unmatched_detections.append(detection)

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            person_id = self._create_new_track(detection, current_time)
            tracked_person = detection.copy()
            tracked_person['person_id'] = person_id
            tracked_person['track_data'] = self.tracks[person_id]
            tracked_people.append(tracked_person)

        return tracked_people

    def clear_all_tracks(self):
        """Force clear all tracks - useful for debugging"""
        if self.tracks:
            print(f"DEBUG: Force clearing {len(self.tracks)} tracks")
        self.tracks.clear()
        self.frames_without_detection = 0

    def get_track_count(self) -> int:
        """Get current number of active tracks"""
        return len(self.tracks)

    def _create_new_track(self, detection: Dict, timestamp: datetime) -> int:
        """Create a new person track"""
        person_id = self.next_id
        self.next_id += 1

        bbox = detection['bbox']
        self.tracks[person_id] = {
            'first_seen': timestamp,
            'last_seen': timestamp,
            'positions': [{
                'x': bbox['center_x'],
                'y': bbox['center_y'],
                'width': bbox['width'],
                'height': bbox['height'],
                'timestamp': timestamp,
                'confidence': detection.get('confidence', 0.5)
            }],
            'behavior_history': [],
            'alert_history': []
        }

        return person_id

    def _update_track(self, person_id: int, detection: Dict, timestamp: datetime):
        """Update an existing person track"""
        bbox = detection['bbox']
        track_data = self.tracks[person_id]

        # Add new position with confidence
        track_data['positions'].append({
            'x': bbox['center_x'],
            'y': bbox['center_y'],
            'width': bbox['width'],
            'height': bbox['height'],
            'timestamp': timestamp,
            'confidence': detection.get('confidence', 0.5)
        })

        # Keep only recent positions (last 100)
        if len(track_data['positions']) > 100:
            track_data['positions'] = track_data['positions'][-100:]

        track_data['last_seen'] = timestamp

    def _remove_expired_tracks(self, current_time: datetime):
        """Remove tracks that haven't been updated recently"""
        expired_ids = []
        timeout_threshold = current_time - \
            timedelta(seconds=self.track_timeout)

        for person_id, track_data in self.tracks.items():
            if track_data['last_seen'] < timeout_threshold:
                expired_ids.append(person_id)

        if expired_ids:
            print(
                f"DEBUG: Removing {len(expired_ids)} expired tracks: {expired_ids}")

        for person_id in expired_ids:
            del self.tracks[person_id]

        # Also remove tracks with consistently low confidence
        low_confidence_ids = []
        for person_id, track_data in self.tracks.items():
            # Check if track has been consistently low confidence
            positions = track_data.get('positions', [])
            if len(positions) >= 3:
                recent_confidences = [pos.get('confidence', 0)
                                      for pos in positions[-3:]]
                avg_confidence = sum(recent_confidences) / \
                    len(recent_confidences)
                if avg_confidence < 0.4:
                    low_confidence_ids.append(person_id)

        for person_id in low_confidence_ids:
            if person_id in self.tracks:
                del self.tracks[person_id]

    def get_track_duration(self, person_id: int) -> Optional[float]:
        """Get how long a person has been tracked (in seconds)"""
        if person_id not in self.tracks:
            return None

        track_data = self.tracks[person_id]
        duration = (track_data['last_seen'] -
                    track_data['first_seen']).total_seconds()
        return duration

    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        This helps ensure we count each person only once, even if there are multiple overlapping detections
        """
        if not detections:
            return detections

        # Filter only high-confidence person detections
        person_detections = [
            d for d in detections
            if d.get('class') == 'person' and d.get('confidence', 0) >= 0.5
        ]

        if len(person_detections) <= 1:
            return person_detections

        # Additional filtering for reasonable person-sized detections
        filtered_detections = []
        for detection in person_detections:
            bbox = detection['bbox']
            width = bbox['width']
            height = bbox['height']

            # Filter out detections that are too small or have unreasonable aspect ratios
            if (width > 30 and height > 50 and  # Minimum size
                height > width and  # People are typically taller than wide
                    height / width < 4):  # Not too thin
                filtered_detections.append(detection)

        if len(filtered_detections) <= 1:
            return filtered_detections

        # Convert to format needed for NMS
        boxes = []
        scores = []

        for detection in filtered_detections:
            bbox = detection['bbox']
            # Convert center coordinates to corner coordinates
            x1 = bbox['center_x'] - bbox['width'] / 2
            y1 = bbox['center_y'] - bbox['height'] / 2
            x2 = bbox['center_x'] + bbox['width'] / 2
            y2 = bbox['center_y'] + bbox['height'] / 2

            boxes.append([x1, y1, x2, y2])
            scores.append(detection.get('confidence', 1.0))

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # Apply NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.5,  # Higher confidence threshold
            nms_threshold=iou_threshold
        )

        # Return filtered detections
        if len(indices) > 0:
            indices = indices.flatten()
            return [filtered_detections[i] for i in indices]
        else:
            return []

    def get_movement_pattern(self, person_id: int, window_seconds: int = 10) -> Dict:
        """Analyze movement pattern for a person"""
        if person_id not in self.tracks:
            return {}

        track_data = self.tracks[person_id]
        positions = track_data['positions']

        if len(positions) < 2:
            return {'total_distance': 0, 'average_speed': 0, 'direction_changes': 0}

        # Filter positions within time window
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=window_seconds)
        recent_positions = [
            pos for pos in positions if pos['timestamp'] >= cutoff_time]

        if len(recent_positions) < 2:
            return {'total_distance': 0, 'average_speed': 0, 'direction_changes': 0}

        # Calculate movement metrics
        total_distance = 0
        direction_changes = 0
        prev_direction = None

        for i in range(1, len(recent_positions)):
            curr_pos = recent_positions[i]
            prev_pos = recent_positions[i-1]

            # Calculate distance
            distance = np.sqrt((curr_pos['x'] - prev_pos['x'])**2 +
                               (curr_pos['y'] - prev_pos['y'])**2)
            total_distance += distance

            # Calculate direction
            if distance > 5:  # Only consider significant movements
                direction = np.arctan2(curr_pos['y'] - prev_pos['y'],
                                       curr_pos['x'] - prev_pos['x'])
                if prev_direction is not None:
                    angle_diff = abs(direction - prev_direction)
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    if angle_diff > np.pi / 4:  # 45 degrees threshold
                        direction_changes += 1
                prev_direction = direction

        # Calculate average speed
        time_span = (recent_positions[-1]['timestamp'] -
                     recent_positions[0]['timestamp']).total_seconds()
        average_speed = total_distance / max(time_span, 1)

        return {
            'total_distance': total_distance,
            'average_speed': average_speed,
            'direction_changes': direction_changes
        }
