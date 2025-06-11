"""
Pose Estimation Component for Shoplifting Detection
Implements REQ-025: Pose estimation using MediaPipe/OpenPose
Analyzes body posture and gestures for suspicious behaviors
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import math

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class PoseKeypoint:
    """Individual pose keypoint with coordinates and confidence"""
    x: float
    y: float
    z: float
    confidence: float


@dataclass
class PoseResult:
    """Complete pose estimation result"""
    keypoints: List[PoseKeypoint]
    pose_confidence: float
    pose_classification: str
    suspicious_gestures: List[str]
    body_angles: Dict[str, float]


class PoseEstimator:
    """
    Advanced pose estimation for shoplifting behavior detection
    Uses MediaPipe Pose for real-time pose estimation
    """
    
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balance between accuracy and speed
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Pose landmark indices for key body parts
        self.landmark_indices = {
            'nose': 0,
            'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        # Thresholds for behavior detection
        self.crouching_threshold = 0.7  # Hip-to-knee ratio
        self.concealment_threshold = 0.3  # Hand-to-body distance
        self.reaching_threshold = 0.4  # Arm extension ratio
        
        logger.info("PoseEstimator initialized with MediaPipe")
    
    def estimate_pose(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Estimate pose from input image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing pose estimation results
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.pose.process(rgb_image)
            
            if results.pose_landmarks:
                # Extract keypoints
                keypoints = self._extract_keypoints(results.pose_landmarks)
                
                # Analyze pose for suspicious behaviors
                pose_analysis = self._analyze_pose_behaviors(keypoints, image.shape)
                
                return {
                    'keypoints': keypoints,
                    'pose_confidence': self._calculate_pose_confidence(keypoints),
                    'pose_classification': pose_analysis['classification'],
                    'suspicious_gestures': pose_analysis['gestures'],
                    'body_angles': pose_analysis['angles'],
                    'behavior_scores': pose_analysis['behavior_scores']
                }
            else:
                return {
                    'keypoints': [],
                    'pose_confidence': 0.0,
                    'pose_classification': 'no_pose_detected',
                    'suspicious_gestures': [],
                    'body_angles': {},
                    'behavior_scores': {}
                }
                
        except Exception as e:
            logger.error(f"Error in pose estimation: {e}")
            return {
                'keypoints': [],
                'pose_confidence': 0.0,
                'pose_classification': 'error',
                'suspicious_gestures': [],
                'body_angles': {},
                'behavior_scores': {}
            }
    
    def _extract_keypoints(self, pose_landmarks) -> List[Dict]:
        """Extract keypoints from MediaPipe pose landmarks"""
        keypoints = []
        
        for i, landmark in enumerate(pose_landmarks.landmark):
            keypoint = {
                'id': i,
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'confidence': landmark.visibility
            }
            keypoints.append(keypoint)
        
        return keypoints
    
    def _analyze_pose_behaviors(self, keypoints: List[Dict], image_shape: Tuple) -> Dict:
        """Analyze pose for suspicious shoplifting behaviors"""
        if not keypoints:
            return {
                'classification': 'no_pose',
                'gestures': [],
                'angles': {},
                'behavior_scores': {}
            }
        
        # Convert keypoints to dictionary for easier access
        kp_dict = {i: kp for i, kp in enumerate(keypoints)}
        
        # Calculate body angles
        body_angles = self._calculate_body_angles(kp_dict)
        
        # Detect suspicious gestures
        gestures = []
        behavior_scores = {}
        
        # 1. Crouching/Bending Detection
        if self._is_crouching(kp_dict, body_angles):
            gestures.append('crouching')
            behavior_scores['crouching'] = 0.7
        
        # 2. Concealment Gestures
        concealment_score = self._detect_concealment_gestures(kp_dict, image_shape)
        if concealment_score > 0.5:
            gestures.append('concealment_gesture')
            behavior_scores['concealment'] = concealment_score
        
        # 3. Reaching/Grabbing Motions
        reaching_score = self._detect_reaching_motions(kp_dict)
        if reaching_score > 0.4:
            gestures.append('reaching')
            behavior_scores['reaching'] = reaching_score
        
        # 4. Suspicious Hand Movements
        hand_score = self._analyze_hand_movements(kp_dict)
        if hand_score > 0.3:
            gestures.append('suspicious_hand_movement')
            behavior_scores['hand_movement'] = hand_score
        
        # 5. Body Orientation Analysis
        orientation_score = self._analyze_body_orientation(kp_dict)
        if orientation_score > 0.4:
            gestures.append('suspicious_orientation')
            behavior_scores['orientation'] = orientation_score
        
        # Classify overall pose
        classification = self._classify_pose(gestures, behavior_scores)
        
        return {
            'classification': classification,
            'gestures': gestures,
            'angles': body_angles,
            'behavior_scores': behavior_scores
        }
    
    def _calculate_body_angles(self, keypoints: Dict) -> Dict[str, float]:
        """Calculate important body angles for behavior analysis"""
        angles = {}
        
        try:
            # Hip-Knee-Ankle angle (for crouching detection)
            if all(idx in keypoints for idx in [23, 25, 27]):  # left hip, knee, ankle
                hip = keypoints[23]
                knee = keypoints[25]
                ankle = keypoints[27]
                angles['left_leg_angle'] = self._calculate_angle(hip, knee, ankle)
            
            if all(idx in keypoints for idx in [24, 26, 28]):  # right hip, knee, ankle
                hip = keypoints[24]
                knee = keypoints[26]
                ankle = keypoints[28]
                angles['right_leg_angle'] = self._calculate_angle(hip, knee, ankle)
            
            # Shoulder-Elbow-Wrist angle (for arm movements)
            if all(idx in keypoints for idx in [11, 13, 15]):  # left shoulder, elbow, wrist
                shoulder = keypoints[11]
                elbow = keypoints[13]
                wrist = keypoints[15]
                angles['left_arm_angle'] = self._calculate_angle(shoulder, elbow, wrist)
            
            if all(idx in keypoints for idx in [12, 14, 16]):  # right shoulder, elbow, wrist
                shoulder = keypoints[12]
                elbow = keypoints[14]
                wrist = keypoints[16]
                angles['right_arm_angle'] = self._calculate_angle(shoulder, elbow, wrist)
            
            # Torso angle (shoulder to hip)
            if all(idx in keypoints for idx in [11, 12, 23, 24]):
                left_shoulder = keypoints[11]
                right_shoulder = keypoints[12]
                left_hip = keypoints[23]
                right_hip = keypoints[24]
                
                shoulder_center = {
                    'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                    'y': (left_shoulder['y'] + right_shoulder['y']) / 2
                }
                hip_center = {
                    'x': (left_hip['x'] + right_hip['x']) / 2,
                    'y': (left_hip['y'] + right_hip['y']) / 2
                }
                
                angles['torso_angle'] = math.degrees(
                    math.atan2(hip_center['y'] - shoulder_center['y'],
                              hip_center['x'] - shoulder_center['x'])
                )
        
        except Exception as e:
            logger.warning(f"Error calculating body angles: {e}")
        
        return angles
    
    def _calculate_angle(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        """Calculate angle between three points"""
        try:
            # Vector from point2 to point1
            v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
            # Vector from point2 to point3
            v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
            angle = math.degrees(math.acos(cos_angle))
            
            return angle
        except:
            return 0.0
    
    def _is_crouching(self, keypoints: Dict, body_angles: Dict) -> bool:
        """Detect if person is crouching/bending"""
        try:
            # Check leg angles
            left_leg_angle = body_angles.get('left_leg_angle', 180)
            right_leg_angle = body_angles.get('right_leg_angle', 180)
            
            # Crouching typically has leg angles < 120 degrees
            avg_leg_angle = (left_leg_angle + right_leg_angle) / 2
            
            # Also check hip-to-shoulder ratio
            if all(idx in keypoints for idx in [11, 12, 23, 24]):
                shoulder_y = (keypoints[11]['y'] + keypoints[12]['y']) / 2
                hip_y = (keypoints[23]['y'] + keypoints[24]['y']) / 2
                
                # If hips are close to shoulders (vertically), person might be crouching
                torso_ratio = abs(hip_y - shoulder_y)
                
                return avg_leg_angle < 120 or torso_ratio < 0.15
            
            return avg_leg_angle < 120
            
        except Exception as e:
            logger.warning(f"Error in crouching detection: {e}")
            return False
    
    def _detect_concealment_gestures(self, keypoints: Dict, image_shape: Tuple) -> float:
        """Detect concealment gestures (hands near body/pockets)"""
        try:
            concealment_score = 0.0
            
            # Check hand positions relative to body
            if all(idx in keypoints for idx in [15, 16, 23, 24]):  # wrists and hips
                left_wrist = keypoints[15]
                right_wrist = keypoints[16]
                left_hip = keypoints[23]
                right_hip = keypoints[24]
                
                # Calculate distances from wrists to hips
                left_distance = math.sqrt(
                    (left_wrist['x'] - left_hip['x'])**2 + 
                    (left_wrist['y'] - left_hip['y'])**2
                )
                right_distance = math.sqrt(
                    (right_wrist['x'] - right_hip['x'])**2 + 
                    (right_wrist['y'] - right_hip['y'])**2
                )
                
                # If hands are very close to hips (pocket area)
                if left_distance < self.concealment_threshold:
                    concealment_score += 0.4
                if right_distance < self.concealment_threshold:
                    concealment_score += 0.4
                
                # Check if hands are behind body (concealment behavior)
                if left_wrist['z'] < left_hip['z'] - 0.1:  # Hand behind hip
                    concealment_score += 0.3
                if right_wrist['z'] < right_hip['z'] - 0.1:
                    concealment_score += 0.3
            
            return min(concealment_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error in concealment detection: {e}")
            return 0.0
    
    def _detect_reaching_motions(self, keypoints: Dict) -> float:
        """Detect reaching/grabbing motions"""
        try:
            reaching_score = 0.0
            
            # Check arm extension
            if all(idx in keypoints for idx in [11, 13, 15]):  # left arm
                shoulder = keypoints[11]
                elbow = keypoints[13]
                wrist = keypoints[15]
                
                # Calculate arm extension ratio
                shoulder_to_wrist = math.sqrt(
                    (wrist['x'] - shoulder['x'])**2 + 
                    (wrist['y'] - shoulder['y'])**2
                )
                shoulder_to_elbow = math.sqrt(
                    (elbow['x'] - shoulder['x'])**2 + 
                    (elbow['y'] - shoulder['y'])**2
                )
                
                if shoulder_to_elbow > 0:
                    extension_ratio = shoulder_to_wrist / shoulder_to_elbow
                    if extension_ratio > 1.5:  # Extended arm
                        reaching_score += 0.4
            
            # Similar check for right arm
            if all(idx in keypoints for idx in [12, 14, 16]):  # right arm
                shoulder = keypoints[12]
                elbow = keypoints[14]
                wrist = keypoints[16]
                
                shoulder_to_wrist = math.sqrt(
                    (wrist['x'] - shoulder['x'])**2 + 
                    (wrist['y'] - shoulder['y'])**2
                )
                shoulder_to_elbow = math.sqrt(
                    (elbow['x'] - shoulder['x'])**2 + 
                    (elbow['y'] - shoulder['y'])**2
                )
                
                if shoulder_to_elbow > 0:
                    extension_ratio = shoulder_to_wrist / shoulder_to_elbow
                    if extension_ratio > 1.5:
                        reaching_score += 0.4
            
            return min(reaching_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error in reaching detection: {e}")
            return 0.0
    
    def _analyze_hand_movements(self, keypoints: Dict) -> float:
        """Analyze suspicious hand movements"""
        # This would require temporal analysis across frames
        # For now, return basic hand position analysis
        try:
            hand_score = 0.0
            
            # Check if hands are in unusual positions
            if all(idx in keypoints for idx in [15, 16]):  # wrists
                left_wrist = keypoints[15]
                right_wrist = keypoints[16]
                
                # Check hand confidence (low confidence might indicate occlusion/concealment)
                if left_wrist['confidence'] < 0.3:
                    hand_score += 0.2
                if right_wrist['confidence'] < 0.3:
                    hand_score += 0.2
            
            return hand_score
            
        except Exception as e:
            logger.warning(f"Error in hand movement analysis: {e}")
            return 0.0
    
    def _analyze_body_orientation(self, keypoints: Dict) -> float:
        """Analyze body orientation for suspicious behavior"""
        try:
            orientation_score = 0.0
            
            # Check if person is facing away (potential concealment)
            if all(idx in keypoints for idx in [0, 11, 12]):  # nose, shoulders
                nose = keypoints[0]
                left_shoulder = keypoints[11]
                right_shoulder = keypoints[12]
                
                # If nose confidence is low but shoulders are visible,
                # person might be facing away
                if nose['confidence'] < 0.3 and \
                   left_shoulder['confidence'] > 0.5 and \
                   right_shoulder['confidence'] > 0.5:
                    orientation_score += 0.5
            
            return orientation_score
            
        except Exception as e:
            logger.warning(f"Error in orientation analysis: {e}")
            return 0.0
    
    def _classify_pose(self, gestures: List[str], behavior_scores: Dict) -> str:
        """Classify overall pose based on detected gestures"""
        if not gestures:
            return 'normal'
        
        # Calculate overall suspicion score
        total_score = sum(behavior_scores.values())
        
        if total_score > 1.5:
            return 'highly_suspicious'
        elif total_score > 0.8:
            return 'suspicious'
        elif total_score > 0.4:
            return 'potentially_suspicious'
        else:
            return 'normal'
    
    def _calculate_pose_confidence(self, keypoints: List[Dict]) -> float:
        """Calculate overall pose confidence"""
        if not keypoints:
            return 0.0
        
        # Average confidence of key body parts
        key_parts = [0, 11, 12, 15, 16, 23, 24]  # nose, shoulders, wrists, hips
        confidences = []
        
        for kp in keypoints:
            if kp['id'] in key_parts:
                confidences.append(kp['confidence'])
        
        return np.mean(confidences) if confidences else 0.0
