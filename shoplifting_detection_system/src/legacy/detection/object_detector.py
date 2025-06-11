import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
from config import Config


class ObjectDetector:
    def __init__(self):
        self.model = YOLO(Config.MODEL_PATH)
        self.target_classes = ['person', 'handbag',
                               'backpack', 'suitcase', 'bottle', 'cell phone']

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in the frame and return detection results
        """
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class name
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    # Only process target classes
                    if class_name in self.target_classes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])

                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': {
                                'x1': int(x1),
                                'y1': int(y1),
                                'x2': int(x2),
                                'y2': int(y2),
                                'width': int(x2 - x1),
                                'height': int(y2 - y1),
                                'center_x': int((x1 + x2) / 2),
                                'center_y': int((y1 + y2) / 2)
                            }
                        }
                        detections.append(detection)

        return detections

    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        """
        Specifically detect people in the frame with strict filtering
        """
        all_detections = self.detect_objects(frame)
        people = []

        for det in all_detections:
            if det['class'] == 'person':
                bbox = det['bbox']
                width = bbox['width']
                height = bbox['height']
                confidence = det['confidence']

                # Apply strict filtering for person detections
                if (confidence >= 0.5 and  # High confidence threshold
                    width >= 30 and height >= 50 and  # Minimum size
                    height > width and  # People are taller than wide
                    height / width <= 4 and  # Not too thin
                        width * height >= 2000):  # Minimum area
                    people.append(det)

        return people

    def detect_items(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect items that could be stolen
        """
        all_detections = self.detect_objects(frame)
        items = [det for det in all_detections if det['class'] != 'person']
        return items

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame
        """
        annotated_frame = frame.copy()

        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']

            # Choose color based on class
            if class_name == 'person':
                color = (0, 255, 0)  # Green for people
            else:
                color = (0, 0, 255)  # Red for items

            # Draw bounding box
            cv2.rectangle(annotated_frame,
                          (bbox['x1'], bbox['y1']),
                          (bbox['x2'], bbox['y2']),
                          color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame,
                          (bbox['x1'], bbox['y1'] - label_size[1] - 10),
                          (bbox['x1'] + label_size[0], bbox['y1']),
                          color, -1)
            cv2.putText(annotated_frame, label,
                        (bbox['x1'], bbox['y1'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated_frame
