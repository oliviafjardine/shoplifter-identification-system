"""
Action Recognition Component for Shoplifting Detection
Implements REQ-025: 3D CNN or Transformer-based action recognition
Analyzes temporal sequences of actions for suspicious behaviors
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque
import logging
from dataclasses import dataclass
import time

from config import Config, DetectionBehavior

logger = logging.getLogger(__name__)


@dataclass
class ActionSequence:
    """Temporal sequence of frames for action recognition"""
    frames: List[np.ndarray]
    timestamps: List[float]
    person_id: int
    sequence_length: int = 16  # Number of frames in sequence


@dataclass
class ActionResult:
    """Action recognition result"""
    action_class: str
    confidence: float
    temporal_features: np.ndarray
    behavior_mapping: Dict[DetectionBehavior, float]


class Simple3DCNN(nn.Module):
    """
    Simplified 3D CNN for action recognition
    In production, this would be replaced with a more sophisticated model
    like I3D, SlowFast, or Video Transformer
    """
    
    def __init__(self, num_classes: int = 8, input_channels: int = 3):
        super(Simple3DCNN, self).__init__()
        
        # 3D Convolutional layers
        self.conv3d1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), 
                                stride=(1, 2, 2), padding=(1, 3, 3))
        self.conv3d2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), 
                                stride=(1, 2, 2), padding=(1, 1, 1))
        self.conv3d3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), 
                                stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv3d4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), 
                                stride=(2, 2, 2), padding=(1, 1, 1))
        
        # Pooling and normalization
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm3d(512)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 1 * 4 * 4, 1024)  # Adjust based on input size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: (batch, channels, depth, height, width)
        x = self.relu(self.bn1(self.conv3d1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv3d2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3d3(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn4(self.conv3d4(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ActionRecognizer:
    """
    Advanced action recognition for shoplifting behavior detection
    Uses temporal analysis of video sequences to identify suspicious actions
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = 16  # Number of frames per sequence
        self.input_size = (112, 112)  # Input frame size
        
        # Action classes for shoplifting detection
        self.action_classes = [
            'normal_shopping',
            'item_concealment',
            'pocket_stuffing',
            'bag_loading',
            'tag_removal',
            'price_switching',
            'coordinated_movement',
            'suspicious_looking'
        ]
        
        # Initialize model
        self.model = self._load_model()
        
        # Frame sequences for each person
        self.person_sequences: Dict[int, Deque[np.ndarray]] = {}
        self.person_timestamps: Dict[int, Deque[float]] = {}
        
        # Action-to-behavior mapping
        self.behavior_mapping = {
            'item_concealment': {
                DetectionBehavior.ITEM_CONCEALMENT: 0.9,
                DetectionBehavior.POCKET_STUFFING: 0.3
            },
            'pocket_stuffing': {
                DetectionBehavior.POCKET_STUFFING: 0.9,
                DetectionBehavior.ITEM_CONCEALMENT: 0.4
            },
            'bag_loading': {
                DetectionBehavior.BAG_LOADING: 0.9,
                DetectionBehavior.ITEM_CONCEALMENT: 0.3
            },
            'tag_removal': {
                DetectionBehavior.SECURITY_TAG_REMOVAL: 0.9
            },
            'price_switching': {
                DetectionBehavior.PRICE_TAG_SWITCHING: 0.9
            },
            'coordinated_movement': {
                DetectionBehavior.COORDINATED_THEFT: 0.8
            },
            'suspicious_looking': {
                DetectionBehavior.ITEM_CONCEALMENT: 0.2,
                DetectionBehavior.POCKET_STUFFING: 0.2
            }
        }
        
        logger.info(f"ActionRecognizer initialized on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load pre-trained action recognition model"""
        try:
            model = Simple3DCNN(num_classes=len(self.action_classes))
            
            # Try to load pre-trained weights
            try:
                model_path = Config.ACTION_MODEL_PATH
                if torch.cuda.is_available():
                    model.load_state_dict(torch.load(model_path))
                else:
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                logger.info(f"Loaded pre-trained action model from {model_path}")
            except:
                logger.warning("Pre-trained action model not found, using random weights")
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading action recognition model: {e}")
            # Return dummy model for testing
            return Simple3DCNN(num_classes=len(self.action_classes)).to(self.device)
    
    def recognize_action(self, person_roi: np.ndarray, person_id: int = 0) -> Dict[str, Any]:
        """
        Recognize action from person region of interest
        
        Args:
            person_roi: Person region of interest from current frame
            person_id: Unique identifier for person tracking
            
        Returns:
            Dictionary containing action recognition results
        """
        try:
            current_time = time.time()
            
            # Initialize sequences for new person
            if person_id not in self.person_sequences:
                self.person_sequences[person_id] = deque(maxlen=self.sequence_length)
                self.person_timestamps[person_id] = deque(maxlen=self.sequence_length)
            
            # Preprocess and add frame to sequence
            processed_frame = self._preprocess_frame(person_roi)
            self.person_sequences[person_id].append(processed_frame)
            self.person_timestamps[person_id].append(current_time)
            
            # Check if we have enough frames for recognition
            if len(self.person_sequences[person_id]) < self.sequence_length:
                return {
                    'action': 'insufficient_frames',
                    'confidence': 0.0,
                    'behavior_scores': {},
                    'temporal_features': np.array([])
                }
            
            # Perform action recognition
            action_result = self._classify_action_sequence(person_id)
            
            return {
                'action': action_result.action_class,
                'confidence': action_result.confidence,
                'behavior_scores': action_result.behavior_mapping,
                'temporal_features': action_result.temporal_features.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in action recognition: {e}")
            return {
                'action': 'error',
                'confidence': 0.0,
                'behavior_scores': {},
                'temporal_features': []
            }
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for action recognition"""
        try:
            # Resize frame
            frame_resized = cv2.resize(frame, self.input_size)
            
            # Normalize pixel values
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2RGB)
            
            return frame_rgb
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return np.zeros((*self.input_size, 3), dtype=np.float32)
    
    def _classify_action_sequence(self, person_id: int) -> ActionResult:
        """Classify action from temporal sequence"""
        try:
            # Get frame sequence
            frames = list(self.person_sequences[person_id])
            
            # Convert to tensor format (batch, channels, depth, height, width)
            sequence_tensor = np.stack(frames, axis=0)  # (depth, height, width, channels)
            sequence_tensor = np.transpose(sequence_tensor, (3, 0, 1, 2))  # (channels, depth, height, width)
            sequence_tensor = np.expand_dims(sequence_tensor, axis=0)  # Add batch dimension
            
            # Convert to PyTorch tensor
            input_tensor = torch.from_numpy(sequence_tensor).float().to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            # Get action class name
            action_class = self.action_classes[predicted_class.item()]
            confidence_score = confidence.item()
            
            # Map to behavior scores
            behavior_scores = self.behavior_mapping.get(action_class, {})
            
            # Extract temporal features (use intermediate layer output)
            temporal_features = self._extract_temporal_features(input_tensor)
            
            return ActionResult(
                action_class=action_class,
                confidence=confidence_score,
                temporal_features=temporal_features,
                behavior_mapping=behavior_scores
            )
            
        except Exception as e:
            logger.error(f"Error classifying action sequence: {e}")
            return ActionResult(
                action_class='error',
                confidence=0.0,
                temporal_features=np.array([]),
                behavior_mapping={}
            )
    
    def _extract_temporal_features(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Extract temporal features from intermediate layers"""
        try:
            # Use features from the last convolutional layer
            with torch.no_grad():
                x = self.model.relu(self.model.bn1(self.model.conv3d1(input_tensor)))
                x = self.model.pool(x)
                x = self.model.relu(self.model.bn2(self.model.conv3d2(x)))
                x = self.model.pool(x)
                x = self.model.relu(self.model.bn3(self.model.conv3d3(x)))
                x = self.model.pool(x)
                features = self.model.relu(self.model.bn4(self.model.conv3d4(x)))
                
                # Global average pooling to get feature vector
                features = torch.mean(features, dim=(2, 3, 4))  # Average over spatial and temporal dimensions
                
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            return np.array([])
    
    def analyze_action_patterns(self, person_id: int, window_size: int = 5) -> Dict[str, Any]:
        """
        Analyze action patterns over a temporal window
        
        Args:
            person_id: Person identifier
            window_size: Number of recent actions to analyze
            
        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            # This would maintain a history of recent actions for each person
            # and analyze patterns like:
            # - Repeated suspicious actions
            # - Escalating suspicious behavior
            # - Action sequences that indicate shoplifting
            
            # For now, return basic analysis
            return {
                'pattern_detected': False,
                'pattern_type': 'none',
                'pattern_confidence': 0.0,
                'escalation_score': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing action patterns: {e}")
            return {
                'pattern_detected': False,
                'pattern_type': 'error',
                'pattern_confidence': 0.0,
                'escalation_score': 0.0
            }
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about action recognition performance"""
        return {
            'active_persons': len(self.person_sequences),
            'total_sequences_processed': sum(len(seq) for seq in self.person_sequences.values()),
            'average_sequence_length': np.mean([len(seq) for seq in self.person_sequences.values()]) if self.person_sequences else 0,
            'model_device': str(self.device),
            'sequence_length': self.sequence_length
        }
    
    def cleanup_old_sequences(self, max_age_seconds: float = 30.0):
        """Clean up old sequences to prevent memory leaks"""
        current_time = time.time()
        persons_to_remove = []
        
        for person_id, timestamps in self.person_timestamps.items():
            if timestamps and (current_time - timestamps[-1]) > max_age_seconds:
                persons_to_remove.append(person_id)
        
        for person_id in persons_to_remove:
            del self.person_sequences[person_id]
            del self.person_timestamps[person_id]
        
        if persons_to_remove:
            logger.info(f"Cleaned up sequences for {len(persons_to_remove)} inactive persons")
    
    def update_behavior_mapping(self, new_mapping: Dict[str, Dict[DetectionBehavior, float]]):
        """Update action-to-behavior mapping (for model fine-tuning)"""
        self.behavior_mapping.update(new_mapping)
        logger.info("Updated behavior mapping for action recognition")
    
    def save_model(self, path: str):
        """Save current model state"""
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Action recognition model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, path: str):
        """Load model from saved state"""
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            logger.info(f"Action recognition model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
