"""
Person Re-Identification Component for Shoplifting Detection
Implements REQ-025: ResNet-based siamese network for person re-ID
Maintains consistent tracking across multiple camera zones
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class PersonFeatures:
    """Person re-identification features"""
    person_id: int
    features: np.ndarray
    timestamp: float
    camera_id: str
    confidence: float
    bounding_box: Dict[str, int]


class ResNetBackbone(nn.Module):
    """ResNet backbone for person re-identification"""
    
    def __init__(self, num_features: int = 2048):
        super(ResNetBackbone, self).__init__()
        
        # Simplified ResNet-like architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks (simplified)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection
        self.fc = nn.Linear(512, num_features)
        self.bn_fc = nn.BatchNorm1d(num_features)
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        
        # First block
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        x = self.bn_fc(x)
        
        # L2 normalization for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        
        return x


class PersonReID:
    """
    Advanced person re-identification system for cross-camera tracking
    Uses deep learning features to maintain consistent person identities
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = (256, 128)  # Height x Width for person images
        self.feature_dim = 2048
        
        # Initialize model
        self.model = self._load_model()
        
        # Person gallery (stored features for known persons)
        self.person_gallery: Dict[int, PersonFeatures] = {}
        self.feature_cache: Dict[str, np.ndarray] = {}  # Cache for recent features
        
        # Re-ID parameters
        self.similarity_threshold = Config.REID_THRESHOLD
        self.max_gallery_size = 1000  # Maximum number of persons to track
        self.feature_update_interval = 10  # Update features every N detections
        
        # Statistics
        self.reid_stats = {
            'total_extractions': 0,
            'successful_matches': 0,
            'new_persons_registered': 0,
            'gallery_size': 0
        }
        
        logger.info(f"PersonReID initialized on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load pre-trained person re-ID model"""
        try:
            model = ResNetBackbone(num_features=self.feature_dim)
            
            # Try to load pre-trained weights
            try:
                model_path = Config.REID_MODEL_PATH
                if torch.cuda.is_available():
                    model.load_state_dict(torch.load(model_path))
                else:
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                logger.info(f"Loaded pre-trained ReID model from {model_path}")
            except:
                logger.warning("Pre-trained ReID model not found, using random weights")
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading person ReID model: {e}")
            return ResNetBackbone(num_features=self.feature_dim).to(self.device)
    
    def extract_features(self, person_image: np.ndarray) -> np.ndarray:
        """
        Extract re-identification features from person image
        
        Args:
            person_image: Person region of interest (BGR format)
            
        Returns:
            Feature vector for person re-identification
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(person_image)
            
            # Extract features
            with torch.no_grad():
                input_tensor = torch.from_numpy(processed_image).unsqueeze(0).to(self.device)
                features = self.model(input_tensor)
                features_np = features.cpu().numpy().flatten()
            
            self.reid_stats['total_extractions'] += 1
            
            return features_np
            
        except Exception as e:
            logger.error(f"Error extracting ReID features: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess person image for feature extraction"""
        try:
            # Resize to standard size
            resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            normalized = rgb_image.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            
            # Convert to CHW format
            processed = np.transpose(normalized, (2, 0, 1))
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return np.zeros((3, self.input_size[0], self.input_size[1]), dtype=np.float32)
    
    def identify_person(self, person_image: np.ndarray, camera_id: str, 
                       bounding_box: Dict[str, int]) -> Dict[str, Any]:
        """
        Identify person and return person ID with confidence
        
        Args:
            person_image: Person region of interest
            camera_id: Camera identifier
            bounding_box: Person bounding box coordinates
            
        Returns:
            Dictionary containing person ID and identification confidence
        """
        try:
            # Extract features
            features = self.extract_features(person_image)
            
            if np.all(features == 0):
                return {
                    'person_id': -1,
                    'confidence': 0.0,
                    'is_new_person': False,
                    'matched_person_id': None
                }
            
            # Search for matching person in gallery
            match_result = self._find_matching_person(features)
            
            if match_result['found']:
                # Update existing person
                person_id = match_result['person_id']
                self._update_person_features(person_id, features, camera_id, bounding_box)
                
                self.reid_stats['successful_matches'] += 1
                
                return {
                    'person_id': person_id,
                    'confidence': match_result['confidence'],
                    'is_new_person': False,
                    'matched_person_id': person_id
                }
            else:
                # Register new person
                new_person_id = self._register_new_person(features, camera_id, bounding_box)
                
                self.reid_stats['new_persons_registered'] += 1
                
                return {
                    'person_id': new_person_id,
                    'confidence': 1.0,
                    'is_new_person': True,
                    'matched_person_id': None
                }
                
        except Exception as e:
            logger.error(f"Error identifying person: {e}")
            return {
                'person_id': -1,
                'confidence': 0.0,
                'is_new_person': False,
                'matched_person_id': None
            }
    
    def _find_matching_person(self, query_features: np.ndarray) -> Dict[str, Any]:
        """Find matching person in gallery using feature similarity"""
        if not self.person_gallery:
            return {'found': False, 'person_id': None, 'confidence': 0.0}
        
        try:
            # Calculate similarities with all persons in gallery
            similarities = []
            person_ids = []
            
            for person_id, person_data in self.person_gallery.items():
                similarity = cosine_similarity(
                    query_features.reshape(1, -1),
                    person_data.features.reshape(1, -1)
                )[0, 0]
                similarities.append(similarity)
                person_ids.append(person_id)
            
            # Find best match
            max_similarity = max(similarities)
            best_match_idx = similarities.index(max_similarity)
            best_person_id = person_ids[best_match_idx]
            
            # Check if similarity exceeds threshold
            if max_similarity >= self.similarity_threshold:
                return {
                    'found': True,
                    'person_id': best_person_id,
                    'confidence': max_similarity
                }
            else:
                return {
                    'found': False,
                    'person_id': None,
                    'confidence': max_similarity
                }
                
        except Exception as e:
            logger.error(f"Error finding matching person: {e}")
            return {'found': False, 'person_id': None, 'confidence': 0.0}
    
    def _register_new_person(self, features: np.ndarray, camera_id: str, 
                           bounding_box: Dict[str, int]) -> int:
        """Register new person in gallery"""
        try:
            # Generate new person ID
            if self.person_gallery:
                new_person_id = max(self.person_gallery.keys()) + 1
            else:
                new_person_id = 1
            
            # Create person features object
            person_features = PersonFeatures(
                person_id=new_person_id,
                features=features,
                timestamp=time.time(),
                camera_id=camera_id,
                confidence=1.0,
                bounding_box=bounding_box
            )
            
            # Add to gallery
            self.person_gallery[new_person_id] = person_features
            
            # Manage gallery size
            self._manage_gallery_size()
            
            self.reid_stats['gallery_size'] = len(self.person_gallery)
            
            return new_person_id
            
        except Exception as e:
            logger.error(f"Error registering new person: {e}")
            return -1
    
    def _update_person_features(self, person_id: int, new_features: np.ndarray, 
                              camera_id: str, bounding_box: Dict[str, int]):
        """Update features for existing person"""
        try:
            if person_id in self.person_gallery:
                person_data = self.person_gallery[person_id]
                
                # Update with exponential moving average
                alpha = 0.1  # Learning rate for feature update
                updated_features = (1 - alpha) * person_data.features + alpha * new_features
                
                # Update person data
                person_data.features = updated_features
                person_data.timestamp = time.time()
                person_data.camera_id = camera_id
                person_data.bounding_box = bounding_box
                
        except Exception as e:
            logger.error(f"Error updating person features: {e}")
    
    def _manage_gallery_size(self):
        """Manage gallery size by removing oldest persons"""
        if len(self.person_gallery) > self.max_gallery_size:
            # Sort by timestamp and remove oldest
            sorted_persons = sorted(
                self.person_gallery.items(),
                key=lambda x: x[1].timestamp
            )
            
            # Remove oldest 10% of persons
            num_to_remove = len(self.person_gallery) // 10
            for i in range(num_to_remove):
                person_id = sorted_persons[i][0]
                del self.person_gallery[person_id]
            
            logger.info(f"Removed {num_to_remove} oldest persons from gallery")
    
    def get_person_trajectory(self, person_id: int) -> List[Dict[str, Any]]:
        """Get trajectory information for a person across cameras"""
        # This would require storing historical location data
        # For now, return basic information
        if person_id in self.person_gallery:
            person_data = self.person_gallery[person_id]
            return [{
                'camera_id': person_data.camera_id,
                'timestamp': person_data.timestamp,
                'bounding_box': person_data.bounding_box
            }]
        return []
    
    def cross_camera_tracking(self, detections: List[Dict]) -> List[Dict]:
        """
        Perform cross-camera tracking for multiple detections
        
        Args:
            detections: List of person detections from different cameras
            
        Returns:
            List of detections with consistent person IDs
        """
        tracked_detections = []
        
        for detection in detections:
            person_image = detection.get('image')
            camera_id = detection.get('camera_id')
            bounding_box = detection.get('bounding_box')
            
            if person_image is not None:
                reid_result = self.identify_person(person_image, camera_id, bounding_box)
                
                detection['person_id'] = reid_result['person_id']
                detection['reid_confidence'] = reid_result['confidence']
                detection['is_new_person'] = reid_result['is_new_person']
                
                tracked_detections.append(detection)
        
        return tracked_detections
    
    def save_gallery(self, filepath: str):
        """Save person gallery to file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.person_gallery, f)
            logger.info(f"Person gallery saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving gallery: {e}")
    
    def load_gallery(self, filepath: str):
        """Load person gallery from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.person_gallery = pickle.load(f)
                self.reid_stats['gallery_size'] = len(self.person_gallery)
                logger.info(f"Person gallery loaded from {filepath}")
            else:
                logger.warning(f"Gallery file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading gallery: {e}")
    
    def get_reid_statistics(self) -> Dict[str, Any]:
        """Get person re-identification statistics"""
        return {
            **self.reid_stats,
            'similarity_threshold': self.similarity_threshold,
            'feature_dimension': self.feature_dim,
            'gallery_size': len(self.person_gallery),
            'device': str(self.device)
        }
    
    def cleanup_old_persons(self, max_age_hours: float = 24.0):
        """Clean up old persons from gallery"""
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        persons_to_remove = []
        for person_id, person_data in self.person_gallery.items():
            if (current_time - person_data.timestamp) > max_age_seconds:
                persons_to_remove.append(person_id)
        
        for person_id in persons_to_remove:
            del self.person_gallery[person_id]
        
        if persons_to_remove:
            logger.info(f"Cleaned up {len(persons_to_remove)} old persons from gallery")
            self.reid_stats['gallery_size'] = len(self.person_gallery)
