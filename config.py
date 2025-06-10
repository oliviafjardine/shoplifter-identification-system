import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://shoplifter_user:shoplifter_pass@localhost:5432/shoplifter_db")
    CAMERA_SOURCE = int(os.getenv("CAMERA_SOURCE", 0))
    ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.7))
    MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Detection thresholds
    CROUCHING_THRESHOLD = 0.6  # Height ratio threshold for crouching detection
    HAND_MOVEMENT_THRESHOLD = 50  # Pixel movement threshold for suspicious hand movements
    LOITERING_TIME_THRESHOLD = 30  # Seconds before considering someone is loitering
    
    # Tracking parameters
    MAX_TRACKING_DISTANCE = 100  # Maximum distance for person tracking
    TRACK_TIMEOUT = 5  # Seconds before losing track of a person
    
    # Alert settings
    ALERT_COOLDOWN = 10  # Seconds between similar alerts for same person
